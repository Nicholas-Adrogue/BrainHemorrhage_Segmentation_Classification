#!/usr/bin/env python3
"""
Hyperparameter Grid Search for Hemorrhage U-Net
================================================
Tests different combinations of Dice weight and BCE foreground weight,
saving the model, predictions, training curves, and a comparison CSV
for each configuration.

Usage:
    cd /Users/Nico/Desktop/MATH7243_ML1/Project
    python grid_search.py

Requires 01_Preprocessed/ and 02_Contour/ to already exist (Steps 1-2).
Edit the GRID below to change which weight combinations are tested.

Results are saved to:
    grid_results/
    ├── dice3_bce10/
    │   ├── model.h5
    │   ├── training_curves.png
    │   ├── prediction_samples/
    │   │   ├── prediction_001.png
    │   │   └── ...
    │   └── history.csv
    ├── dice5_bce10/
    │   └── ...
    └── summary.csv          ← comparison of all runs
"""

# ── Import everything from the main pipeline ─────────────────────────────────
import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import the main script's functions by adding its directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from hemorrhage_segmentation_unet import (
    step1_build_arrays, step2_split_and_augment,
    build_unet, MyMeanIOU, make_tf_datasets,
    create_mask, show_predictions,
    PATCH_SIZE, CHANNELS, OUTPUT_CLASSES, BATCH_SIZE,
    BUFFER_SIZE, EPOCHS, PATIENCE, MIN_DELTA, IMG_SIZE,
    RANDOM_SEED,
)

# Seeds are already set by the import above, but be explicit
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  GRID CONFIGURATION — edit these to test different combinations              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

GRID = [
    # (dice_weight, bce_foreground_weight)
    (1.0,   5.0),
    (1.0,  10.0),
    (1.0,  20.0),
    (3.0,   5.0),
    (3.0,  10.0),
    (3.0,  20.0),
    (5.0,   5.0),
    (5.0,  10.0),
    (5.0,  20.0),
    (10.0, 10.0),
]

GRID_DIR   = os.path.join(BASE_DIR, 'grid_results')
NUM_PREDS  = 10   # how many prediction images to save per run

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  GRID SEARCH                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def make_loss_fn(dice_weight, bce_fg_weight):
    """Create a Dice+BCE loss function with the given weights."""
    def loss_fn(y_true, y_pred):
        probs   = tf.nn.softmax(y_pred, axis=-1)
        pred_fg = probs[..., 1]
        true_fg = tf.cast(tf.squeeze(y_true, axis=-1), tf.float32)

        # Dice
        smooth = 1.0
        intersection = tf.reduce_sum(pred_fg * true_fg)
        dice = (2.0 * intersection + smooth) / (
            tf.reduce_sum(pred_fg) + tf.reduce_sum(true_fg) + smooth
        )
        dice_loss = 1.0 - dice

        # Weighted cross-entropy
        sample_weights = tf.where(true_fg > 0.5, bce_fg_weight, 1.0)
        ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )(y_true, y_pred)
        weighted_ce = tf.reduce_mean(ce * sample_weights)

        return dice_weight * dice_loss + weighted_ce

    loss_fn.__name__ = f'dice{dice_weight}_bce{bce_fg_weight}'
    return loss_fn


def run_one(dice_w, bce_w, ds_train, ds_val, ds_test,
            X_train, steps_per_epoch):
    """Train a single model with the given weights and return metrics."""
    tag = f'dice{dice_w}_bce{bce_w}'.replace('.', 'p')
    run_dir = os.path.join(GRID_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'  Dice weight = {dice_w}   |   BCE foreground weight = {bce_w}')
    print(f'  Saving to {run_dir}/')
    print(f'{"="*70}\n')

    # Build fresh model
    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)
    keras.backend.clear_session()
    model = build_unet(input_shape, OUTPUT_CLASSES)

    loss_fn = make_loss_fn(dice_w, bce_w)
    miou = MyMeanIOU(num_classes=OUTPUT_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=['accuracy', miou],
    )

    checkpoint_path = os.path.join(run_dir, 'model.h5')
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True,
                monitor='val_my_mean_iou', mode='max',
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_my_mean_iou', mode='max',
                min_delta=MIN_DELTA, patience=PATIENCE,
            ),
        ],
    )

    # Reload best weights
    model.load_weights(checkpoint_path)

    # Evaluate on test set
    print('\n── Test-set evaluation ──')
    results = model.evaluate(ds_test, return_dict=True)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(run_dir, 'history.csv'), index_label='epoch')

    # Save training curves
    h = history.history
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.plot(h['loss'], label='Train')
    ax1.plot(h['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(h['accuracy'], label='Train')
    ax2.plot(h['val_accuracy'], label='Val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    miou_key = [k for k in h if 'mean_iou' in k and 'val' not in k][0]
    val_miou_key = 'val_' + miou_key
    ax3.plot(h[miou_key], label='Train')
    ax3.plot(h[val_miou_key], label='Val')
    ax3.set_title('Mean IoU')
    ax3.set_xlabel('Epoch')
    ax3.legend()

    plt.suptitle(f'Dice weight = {dice_w}, BCE fg weight = {bce_w}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_curves.png'), dpi=150)
    plt.close(fig)

    # Save prediction images
    pred_dir = os.path.join(run_dir, 'prediction_samples')
    show_predictions(model, ds_test, num=NUM_PREDS, save_dir=pred_dir)

    # Extract best metrics
    best_epoch     = int(np.argmin(h['val_loss']))
    best_val_loss  = h['val_loss'][best_epoch]
    best_val_acc   = h['val_accuracy'][best_epoch]
    best_val_miou  = h[val_miou_key][best_epoch]
    test_loss      = results.get('loss', float('nan'))
    test_acc       = results.get('accuracy', float('nan'))
    test_miou_key  = [k for k in results if 'mean_iou' in k][0] if any('mean_iou' in k for k in results) else None
    test_miou      = results.get(test_miou_key, float('nan')) if test_miou_key else float('nan')

    return {
        'dice_weight':     dice_w,
        'bce_fg_weight':   bce_w,
        'best_epoch':      best_epoch + 1,
        'total_epochs':    len(h['loss']),
        'best_val_loss':   round(best_val_loss, 4),
        'best_val_acc':    round(best_val_acc, 4),
        'best_val_miou':   round(best_val_miou, 4),
        'test_loss':       round(test_loss, 4),
        'test_accuracy':   round(test_acc, 4),
        'test_miou':       round(test_miou, 4),
        'model_path':      checkpoint_path,
    }


def main():
    os.makedirs(GRID_DIR, exist_ok=True)

    print('══ Loading data (Steps 3-4) ══')
    scans, masks, labels, display_labels = step1_build_arrays()
    (X_train, M_train, y_train,
     X_val, M_val, y_val,
     X_test, M_test, y_test) = step2_split_and_augment(scans, masks, labels, display_labels)

    # Build tf.data datasets once (shared across all runs)
    ds_train, ds_val, ds_test = make_tf_datasets(
        X_train, M_train, X_val, M_val, X_test, M_test
    )
    steps_per_epoch = max(1, X_train.shape[0] // BATCH_SIZE)

    print(f'\n══ Grid search: {len(GRID)} configurations ══')
    print(f'   Each trains for up to {EPOCHS} epochs')
    print(f'   Results → {GRID_DIR}/\n')

    # Run each configuration
    all_results = []
    for i, (dice_w, bce_w) in enumerate(GRID):
        print(f'\n[{i+1}/{len(GRID)}]')
        result = run_one(dice_w, bce_w, ds_train, ds_val, ds_test,
                         X_train, steps_per_epoch)
        all_results.append(result)

        # Save running summary after each run (in case of interruption)
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(os.path.join(GRID_DIR, 'summary.csv'), index=False)
        print(f'\n  Updated summary.csv ({len(all_results)} runs complete)')

    # Print final summary
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('test_miou', ascending=False)
    summary_df.to_csv(os.path.join(GRID_DIR, 'summary.csv'), index=False)

    print('\n' + '='*70)
    print('  GRID SEARCH COMPLETE — Results ranked by test MeanIoU')
    print('='*70)
    print(summary_df[['dice_weight', 'bce_fg_weight', 'best_val_miou',
                       'test_miou', 'test_accuracy', 'total_epochs']].to_string(index=False))
    print(f'\nFull results saved to {GRID_DIR}/summary.csv')
    print(f'Best config: Dice={summary_df.iloc[0]["dice_weight"]}, '
          f'BCE={summary_df.iloc[0]["bce_fg_weight"]} '
          f'(test MeanIoU = {summary_df.iloc[0]["test_miou"]})')


if __name__ == '__main__':
    main()
