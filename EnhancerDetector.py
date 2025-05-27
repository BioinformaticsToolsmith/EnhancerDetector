#!/usr/bin/env python
# coding: utf-8

# ## The purpose of this notebook is to be the final version of EnhancerDetector for publication
# 
# ### This notebook will take an input fasta file of 400 base pair length and output their probability of being a enhancer
# ### Optional, output will also have a Class Activation map of the sequences.

# ### @author: Luis Solis, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# ### @author: Dr. Hani Z. Girgis, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# 
# #### Date Created: 05-27-2025

import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from Nets import CustomConvLayer
from Metrics import weighted_f1_score, specificity
from OneNucleotideIndexer import OneNucleotideIndexer

from Bio import SeqIO
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys


parser = argparse.ArgumentParser(description="EnhancerDetector - Classify sequences as enhancers and optionally generate CAM visualizations.")
parser.add_argument("--species", required=True, choices=["human", "mouse", "fly"], help="Species model to use.")
parser.add_argument("--input", required=True, help="Input FASTA file with sequences.")
parser.add_argument("--cam", action="store_true", help="Enable CAM heatmap output.")
parser.add_argument("--outdir", default="Output", help="Directory to save output files.")
args = parser.parse_args()

species_max_len = {"human": 400, "mouse": 400, "fly": 500}
max_len = species_max_len[args.species]

species_configs = {
    "human": {
        "model": "Models/Human/Single_Classifier_64_3_20.keras",
        "indexer": "Models/Human/indexer.pkl"
    },
    "fly": {
        "model": "Models/Fly/Single_Classifier_32_3_20.keras",
        "indexer": "Models/Fly/indexer_fly.pkl"
    },
    "mouse": {
        "model": "Models/Mouse/FineTune_Classifier_64_3_20_No_Convolution.keras",
        "indexer": "Models/Mouse/indexer_human.pkl"
    }
}



output_cam_pdf = True if args.cam else False

# ### Parse input files and grab their names for output and CAM
# ### Encode the input sequences

similar_seq_list = list(SeqIO.parse(args.input, "fasta"))

similar_name_list = []

for seq in similar_seq_list:
    similar_name_list.append(seq.id)

'''
Input will include all sequences that will be tested to see if they are an enhancer or not, this must be in fasta format
Output will include the sequence id and their probability of being an enhancer.
'''

# Fly model uses ensemble, load all required models and encoders
if args.species == "fly":
    fly_model_paths = [
        "Models/Fly/Single_Classifier_32_3_20.keras",
        "Models/Fly/Single_Classifier_40_3_20.keras",
        "Models/Fly/FineTune_Classifier_64_3_20_With_No_Convolution_0.h5"
    ]
    fly_models = [load_model(path, custom_objects={'CustomConvLayer': CustomConvLayer, 'specificity': specificity, 'weighted_f1_score': weighted_f1_score}) for path in fly_model_paths]
    with open("Models/Fly/indexer_fly.pkl", 'rb') as f:
        fly_indexer = pickle.load(f)
    with open("Models/Fly/indexer_human.pkl", 'rb') as f:
        human_indexer = pickle.load(f)
    human_indexer = OneNucleotideIndexer(max_len, human_indexer)
    matrix_fly = fly_indexer.encode_list(similar_seq_list)
    matrix_human = human_indexer.encode_list(similar_seq_list)
else:
    model_path = {
        "human": "Models/Human/Single_Classifier_64_3_20.keras",
        "mouse": "Models/Mouse/FineTune_Classifier_64_3_20_No_Convolution.keras"
    }[args.species]
    indexer_path = {
        "human": "Models/Human/indexer.pkl",
        "mouse": "Models/Mouse/indexer_human.pkl"
    }[args.species]
    model = load_model(model_path, custom_objects={'CustomConvLayer': CustomConvLayer, 'specificity': specificity, 'weighted_f1_score': weighted_f1_score})
    with open(indexer_path, 'rb') as f:
        indexer = pickle.load(f)
    matrix = indexer.encode_list(similar_seq_list)

# Create tensor
batch_size = len(similar_seq_list)
tensor = np.zeros((batch_size, 1, max_len, 1), dtype=np.int8)
if args.species == "fly":
    tensor_fly = np.zeros_like(tensor)
    for i in range(batch_size):
        tensor_fly[i, 0, :, 0] = matrix_fly[i]
        tensor[i, 0, :, 0] = matrix_human[i]  # for finetuned model
else:
    for i in range(batch_size):
        tensor[i, 0, :, 0] = matrix[i]

# Prediction
if args.species == "fly":
    preds = [m.predict(tensor_fly if i < 2 else tensor) for i, m in enumerate(fly_models)]
    output_prediction = np.mean(preds, axis=0)
    reference_model = fly_models[0]
else:
    output_prediction = model.predict(tensor)
    reference_model = model

# Write predictions
with open(f"{args.outdir}/Model_Output.txt", 'w') as file:
    for name, prob in zip(similar_name_list, output_prediction):
        file.write(f"{name} {prob[0]:.2f}\n")


# ### Below is code for making the CAM model
# ### Cam model was based on code from Deep Learning with Python by Francois Chollet

if args.cam:
    input_tensor = reference_model.input
    last_conv_layer = reference_model.get_layer("custom_conv_layer_3")
    cam_model = Model(inputs=input_tensor, outputs=last_conv_layer.output)
    first_dense_layer = reference_model.get_layer("fc_layer_1")
    class_model = Model(inputs=first_dense_layer.input, outputs=reference_model.output)

    def plot_CAM_map(heatmap_interpolated_list, output_dir, name_list, save_pdf):
        num_sequences = len(heatmap_interpolated_list)
        if num_sequences == 0:
            print("No heatmaps to plot.")
            return

        fig, axs = plt.subplots(num_sequences, 1, figsize=(8.5, 2 * num_sequences))
        if num_sequences == 1:
            axs = [axs]

        for i, heatmap_interpolated in enumerate(heatmap_interpolated_list):
            image = axs[i].matshow(
                heatmap_interpolated.reshape(1, -1),
                cmap='jet', aspect='auto', vmin=0, vmax=1
            )
            axs[i].set_yticks([])
            axs[i].xaxis.set_ticks_position('bottom')
            axs[i].set_xlim(-0.5, len(heatmap_interpolated))
            axs[i].set_title(f'Seq: {name_list}', fontsize=10)
            axs[i].set_xlabel('Nucleotide position')
            if i != num_sequences - 1:
                axs[i].set_xticks([])
            for spine in axs[i].spines.values():
                spine.set_visible(False)

        fig.colorbar(image, ax=axs, orientation='vertical', fraction=0.025, pad=0.02)
        if save_pdf:
            plt.savefig(f'{output_dir}.pdf')
        plt.close(fig)

    def calculate_cam(x_batch_sample):
        with tf.GradientTape() as tape:
            cam_output = cam_model(x_batch_sample, training=False)
            cam_output_flattened = tf.reshape(cam_output, (1, -1))
            tape.watch(cam_output)
            preds = class_model(cam_output_flattened, training=False)
            target_class_pred = preds[0]

        grads = tape.gradient(target_class_pred, cam_output)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        cam_output = cam_output[0]
        heatmap = cam_output * pooled_grads[0]
        heatmap = tf.reduce_mean(heatmap, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / tf.reduce_max(heatmap)
        return heatmap.numpy()

    # Generate CAMs
    for i in range(batch_size):
        x_input = (tensor_fly if args.species == "fly" else tensor)[i:i+1]
        heatmap = calculate_cam(x_input)
        heatmap = heatmap.flatten()
        old_indices = np.linspace(0, heatmap.shape[0] - 1, num=heatmap.shape[0])
        new_indices = np.linspace(0, heatmap.shape[0] - 1, num=max_len)
        heatmap_interpolated = np.interp(new_indices, old_indices, heatmap)
        name = similar_name_list[i]
        plot_CAM_map([heatmap_interpolated], f'{args.outdir}/{name}_CAM', name, save_pdf=True)






