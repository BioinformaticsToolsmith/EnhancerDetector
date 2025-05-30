#!/usr/bin/env python3
# coding: utf-8
# ## The purpose of this notebook is to be the final version of EnhancerDetector for publication
# 
# ### This notebook will take an input enhancer fasta file of 400 base pair length and a control fasta file of 400 base pair length and output a new finetuned enhancer detector model based on the given enhancers.
# ### Recommended to use 20,000 enhancers and 40,000 control sequences. Keep a 2:1 ratio of control to enhancers.

# ### @author: Luis Solis, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# ### @author: Dr. Hani Z. Girgis, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# 
# #### Date Created: 05-29-2025


import argparse
import gc
import os
import random
import pickle
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from Loaders import CombinedControlEnhancerLoader
from Nets import CustomConvLayer, make_single_conv_model_from_model
from Metrics import specificity, weighted_f1_score

def shuffle_sequence(seq, k=1):
    chars = [seq[i:i + k] for i in range(0, len(seq), k)]
    random.shuffle(chars)
    shuffled_seq = ''.join(chars)
    if shuffled_seq == seq:
        raise ValueError("Shuffled sequence is identical to the original.")
    return shuffled_seq

def load_sequences(enhancer_fasta, control_fasta, use_shuffled_controls=False, shuffle_fraction=0.2):
    enhancer_seqs = [str(rec.seq).upper() for rec in SeqIO.parse(enhancer_fasta, "fasta")]
    control_seqs = [str(rec.seq).upper() for rec in SeqIO.parse(control_fasta, "fasta")]

    for seq in enhancer_seqs + control_seqs:
        assert len(seq) == 400, "All sequences must be 400bp"

    expected_controls = 2 * len(enhancer_seqs)
    if len(control_seqs) > expected_controls:
        control_seqs = random.sample(control_seqs, expected_controls)
    elif len(control_seqs) < expected_controls:
        raise ValueError(f"Too few controls ({len(control_seqs)}), expected at least {expected_controls}")

    if use_shuffled_controls:
        num_replace = int(shuffle_fraction * len(control_seqs))
        shuffled_controls = []
        for seq in random.sample(enhancer_seqs, num_replace):
            try:
                shuffled_controls.append(shuffle_sequence(seq, k=random.randint(1, 6)))
            except ValueError:
                continue
        control_seqs[-num_replace:] = shuffled_controls

    return enhancer_seqs, control_seqs

def freeze_layers_except(model, layer_names_to_keep):
    for layer in model.layers:
        if layer.name in layer_names_to_keep:
            layer.trainable = True
            if 'custom_conv_layer' in layer.name:
                layer.unfreeze_layers()
        else:
            if 'custom_conv_layer' in layer.name:
                layer.freeze_layers()
            layer.trainable = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhancers", required=True, help="Path to enhancer FASTA file")
    parser.add_argument("--controls", required=True, help="Path to control FASTA file")
    parser.add_argument("--output_dir", required=True, help="Output prefix/path")
    parser.add_argument("--model", default="Models/Human/Single_Classifier_64_3_20.keras", help="Pretrained model path")
    parser.add_argument("--indexer", default="Models/Human/indexer.pkl", help="Indexer pickle file")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_shuffle", action="store_true", help="Use shuffled enhancers as some controls")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load indexer
    with open(args.indexer, "rb") as f:
        indexer = pickle.load(f)

    enhancer_seqs, control_seqs = load_sequences(args.enhancers, args.controls, use_shuffled_controls=args.use_shuffle)

    enhancer_records = [SeqRecord(Seq(seq), id=f"enh_{i}") for i, seq in enumerate(enhancer_seqs)]
    control_records = [SeqRecord(Seq(seq), id=f"ctrl_{i}") for i, seq in enumerate(control_seqs)]

    combined_records = enhancer_records + control_records
    f_matrix = indexer.encode_list(combined_records)
    rc_matrix = indexer.encode_list([rec.reverse_complement(id="RC_" + rec.id, description="RC") for rec in combined_records])

    enhancer_indices = np.arange(len(enhancer_records))
    control_indices = np.arange(len(enhancer_records), len(combined_records))

    np.save(os.path.join(args.output_dir, "f_matrix.npy"), f_matrix)
    np.save(os.path.join(args.output_dir, "rc_matrix.npy"), rc_matrix)
    np.save(os.path.join(args.output_dir, "enhancer_index.npy"), enhancer_indices)
    np.save(os.path.join(args.output_dir, "control_index.npy"), control_indices)

    # Split indices
    enh_train, enh_val = train_test_split(enhancer_indices, test_size=0.2, random_state=42)
    ctl_train, ctl_val = train_test_split(control_indices, test_size=0.2, random_state=42)

    np.save(os.path.join(args.output_dir, "train_enhancer_index.npy"), enh_train)
    np.save(os.path.join(args.output_dir, "val_enhancer_index.npy"), enh_val)
    np.save(os.path.join(args.output_dir, "train_control_index.npy"), ctl_train)
    np.save(os.path.join(args.output_dir, "val_control_index.npy"), ctl_val)

    train_enh_idx = os.path.join(args.output_dir, "train_enhancer_index.npy")
    val_enh_idx   = os.path.join(args.output_dir, "val_enhancer_index.npy")
    train_ctl_idx = os.path.join(args.output_dir, "train_control_index.npy")
    val_ctl_idx   = os.path.join(args.output_dir, "val_control_index.npy")

    # Loaders
    train_loader = CombinedControlEnhancerLoader(f_matrix, rc_matrix,
        [train_enh_idx], [train_ctl_idx], args.batch_size)
    val_loader = CombinedControlEnhancerLoader(f_matrix, rc_matrix,
        [val_enh_idx], [val_ctl_idx], args.batch_size)

    del f_matrix, rc_matrix
    gc.collect()

    model = load_model(args.model, custom_objects={
        'CustomConvLayer': CustomConvLayer,
        'specificity': specificity,
        'weighted_f1_score': weighted_f1_score
    })

    hybrid = make_single_conv_model_from_model(model, 400, filter_num=64, filter_size=3, unit_num=20, vocab_size=indexer.get_vocabulary_size())
    hybrid.compile(optimizer=Adam(1e-5), loss='binary_crossentropy',
                   metrics=['accuracy', tf.keras.metrics.Recall(name='recall'),
                            specificity, tf.keras.metrics.Precision(name='precision'),
                            weighted_f1_score])

    freeze_layers_except(hybrid, ['output_layer', 'fc_layer_2', 'fc_layer_1', 'relu_2', 'bn_fc_layer_2', 'relu_1', 'bn_fc_layer_1'])

    early_stop = keras.callbacks.EarlyStopping(patience=5, min_delta=1e-5, restore_best_weights=True,
                                               monitor='val_weighted_f1_score', mode='max')

    hybrid.fit(train_loader, epochs=500, validation_data=val_loader, callbacks=[early_stop], verbose=2)

    hybrid.save(os.path.join(args.output_dir, "model_finetuned.keras"))

if __name__ == "__main__":
    main()
