import torch
import torch.nn as nn
from scipy.stats import ttest_ind, mannwhitneyu
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import os
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
from data_processing import TextureDataset, load_texture_tiles
from train import load_model


def create_distorted_images(image, distortion_type='sun_flare', intensity=1.0, save_path=None):
    if len(image.shape) == 2:  # grayscale input
        image_3ch = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image_3ch = (image * 255).astype(np.uint8)

    # distortion types
    if distortion_type == 'sun_flare':
        transform = A.Compose([
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                src_radius=int(120 * intensity),
                src_color=(255, 255, 255),
                p=1.0
            )
        ])
    elif distortion_type == 'grid_shuffle':
        transform = A.Compose([
            A.GridDropout(
                ratio=0.4 * intensity,
                random_offset=True,
                p=1.0
            )
        ])
    elif distortion_type == 'spatter':
        transform = A.Compose([
            A.Spatter(
                mean=0.65 * intensity,
                std=0.3 * intensity,
                cutout_threshold=0.68,
                intensity=0.6 * intensity,
                mode='rain',
                p=1.0
            )
        ])
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")

    try:
        distorted = transform(image=image_3ch)['image']

        if len(distorted.shape) == 3:
            distorted = cv2.cvtColor(distorted, cv2.COLOR_RGB2GRAY)

        distorted = distorted.astype(np.float32) / 255.0

        if save_path:
            cv2.imwrite(save_path, (distorted * 255).astype(np.uint8))

        return distorted

    except Exception as e:
        print(f"Error applying {distortion_type}: {e}")
        return image



def evaluate_model(model, test_loader, device, criterion=None):
    if criterion is None:
        criterion = nn.MSELoss()

    model.eval()
    total_loss = 0
    reconstruction_errors = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # batch loss
            batch_loss = criterion(output, target)
            total_loss += batch_loss.item()

            # per-sample reconstruction errors
            batch_errors = torch.mean((output - target) ** 2, dim=[1, 2, 3])
            reconstruction_errors.extend(batch_errors.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    return {
        'avg_loss': avg_loss,
        'reconstruction_errors': np.array(reconstruction_errors)
    }


def evaluate_anomaly_detection(model, test_images, device, distortion_types=['sun_flare', 'grid_shuffle', 'spatter'],
                               intensities=[0.5, 1.0, 1.5], num_samples=50):
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    results = {
        'normal_errors': [],
        'anomaly_errors': {},
        'statistical_tests': {},
        'roc_metrics': {}
    }

    # random sample test images
    test_sample = np.random.choice(test_images, min(num_samples, len(test_images)), replace=False)

    print("Evaluating normal (undistorted) images...")
    # evaluate normal images
    for img_path in test_sample:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0

        # convert to tensor
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            reconstruction = model(img_tensor)
            error = criterion(reconstruction, img_tensor)
            mse = torch.mean(error).item()
            results['normal_errors'].append(mse)

    # evaluate anomalous (distorted) images
    for distortion_type in distortion_types:
        print(f"Evaluating {distortion_type} distortions...")
        results['anomaly_errors'][distortion_type] = {}

        for intensity in intensities:
            errors = []

            for img_path in test_sample:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0

                # apply distortion
                distorted_img = create_distorted_images(img, distortion_type, intensity)

                if img_path == test_sample[0] and intensity in [0.5, 1.0, 1.5]:
                    save_dir = os.path.join('test_results', 'distorted_samples', distortion_type)
                    os.makedirs(save_dir, exist_ok=True)
                    save_name = f"{distortion_type}_intensity_{intensity}.png"
                    save_path = os.path.join(save_dir, save_name)
                    _ = create_distorted_images(img, distortion_type, intensity, save_path=save_path)

                img_tensor = torch.tensor(distorted_img).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    reconstruction = model(img_tensor)
                    error = criterion(reconstruction, img_tensor)
                    mse = torch.mean(error).item()
                    errors.append(mse)

            results['anomaly_errors'][distortion_type][intensity] = errors

    # analysis
    print("Performing statistical analysis...")
    normal_errors = np.array(results['normal_errors'])

    for distortion_type in distortion_types:
        results['statistical_tests'][distortion_type] = {}
        results['roc_metrics'][distortion_type] = {}

        for intensity in intensities:
            anomaly_errors = np.array(results['anomaly_errors'][distortion_type][intensity])

            # Statistical tests
            # 1. T-test
            t_stat, t_pvalue = ttest_ind(normal_errors, anomaly_errors)

            # 2. Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = mannwhitneyu(normal_errors, anomaly_errors, alternative='two-sided')

            # 3. Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(normal_errors) - 1) * np.var(normal_errors, ddof=1) +
                                  (len(anomaly_errors) - 1) * np.var(anomaly_errors, ddof=1)) /
                                 (len(normal_errors) + len(anomaly_errors) - 2))
            cohens_d = (np.mean(anomaly_errors) - np.mean(normal_errors)) / pooled_std

            results['statistical_tests'][distortion_type][intensity] = {
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'mannwhitney_u': u_stat,
                'mannwhitney_pvalue': u_pvalue,
                'cohens_d': cohens_d,
                'normal_mean': np.mean(normal_errors),
                'normal_std': np.std(normal_errors),
                'anomaly_mean': np.mean(anomaly_errors),
                'anomaly_std': np.std(anomaly_errors)
            }

            # ROC analysis
            # create labels (0 for normal, 1 for anomaly)
            labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
            scores = np.concatenate([normal_errors, anomaly_errors])

            # ROC AUC
            try:
                auc_score = roc_auc_score(labels, scores)
                fpr, tpr, thresholds = roc_curve(labels, scores)
                youden_index = tpr - fpr
                optimal_threshold_idx = np.argmax(youden_index)
                optimal_threshold = thresholds[optimal_threshold_idx]

                results['roc_metrics'][distortion_type][intensity] = {
                    'auc_score': auc_score,
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'optimal_threshold': optimal_threshold,
                    'optimal_fpr': fpr[optimal_threshold_idx],
                    'optimal_tpr': tpr[optimal_threshold_idx]
                }
            except ValueError as e:
                print(f"ROC calculation failed for {distortion_type} at intensity {intensity}: {e}")
                results['roc_metrics'][distortion_type][intensity] = None

    return results


def plot_anomaly_analysis(results, save_dir='anomaly_analysis'):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reconstruction Error Distributions: Normal vs Anomalous', fontsize=16)

    normal_errors = results['normal_errors']
    distortion_types = list(results['anomaly_errors'].keys())

    for idx, distortion_type in enumerate(distortion_types):
        if idx >= 3:
            break

        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        ax.hist(normal_errors, bins=20, alpha=0.7, label='Normal', color='blue', density=True)

        colors = ['red', 'orange', 'darkred']
        intensities = [0.5, 1.0, 1.5]

        for i, intensity in enumerate(intensities):
            if intensity in results['anomaly_errors'][distortion_type]:
                anomaly_errors = results['anomaly_errors'][distortion_type][intensity]
                ax.hist(anomaly_errors, bins=20, alpha=0.5,
                        label=f'{distortion_type.replace("_", " ").title()} ({intensity})',
                        color=colors[i], density=True)

        ax.set_xlabel('Reconstruction Error (MSE)')
        ax.set_ylabel('Density')
        ax.set_title(f'{distortion_type.replace("_", " ").title()} Distortion')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    # heatmap
    distortion_names = []
    p_values = []
    effect_sizes = []

    for distortion_type in results['statistical_tests']:
        for intensity in results['statistical_tests'][distortion_type]:
            stats_data = results['statistical_tests'][distortion_type][intensity]
            distortion_names.append(f"{distortion_type.replace('_', ' ').title()} ({intensity})")
            p_values.append(stats_data['t_pvalue'])
            effect_sizes.append(abs(stats_data['cohens_d']))

    significance_levels = []
    for p_val in p_values:
        if p_val < 0.001:
            significance_levels.append('***')
        elif p_val < 0.01:
            significance_levels.append('**')
        elif p_val < 0.05:
            significance_levels.append('*')
        else:
            significance_levels.append('ns')

    bars = ax.bar(range(len(distortion_names)), effect_sizes,
                  color=['red' if p < 0.05 else 'gray' for p in p_values])

    for i, (bar, sig) in enumerate(zip(bars, significance_levels)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                sig, ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Distortion Type and Intensity')
    ax.set_ylabel('Effect Size (|Cohen\'s d|)')
    ax.set_title(
        'Statistical Significance of Anomaly Detection\n(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)')
    ax.set_xticks(range(len(distortion_names)))
    ax.set_xticklabels(distortion_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistical_significance.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # ROC Curves
    fig, axes = plt.subplots(1, len(distortion_types), figsize=(5 * len(distortion_types), 5))
    if len(distortion_types) == 1:
        axes = [axes]

    for idx, distortion_type in enumerate(distortion_types):
        ax = axes[idx]

        intensities = [0.5, 1.0, 1.5]
        colors = ['red', 'orange', 'darkred']

        for i, intensity in enumerate(intensities):
            if (intensity in results['roc_metrics'][distortion_type] and
                    results['roc_metrics'][distortion_type][intensity] is not None):
                roc_data = results['roc_metrics'][distortion_type][intensity]
                fpr, tpr = roc_data['fpr'], roc_data['tpr']
                auc_score = roc_data['auc_score']

                ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f'Intensity {intensity} (AUC = {auc_score:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {distortion_type.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_reconstructions(model, test_images, device, num_samples=6, save_path=None):
    model.eval()

    # random samples
    sample_paths = np.random.choice(test_images, num_samples, replace=False)

    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    with torch.no_grad():
        for i, img_path in enumerate(sample_paths):
            # original image
            original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            original = original.astype(np.float32) / 255.0

            # distorted version
            distorted = create_distorted_images(original, 'sun_flare', 1.0)

            # get reconstructions
            original_tensor = torch.tensor(original).unsqueeze(0).unsqueeze(0).to(device)
            distorted_tensor = torch.tensor(distorted).unsqueeze(0).unsqueeze(0).to(device)

            original_recon = model(original_tensor).cpu().numpy()[0, 0]
            distorted_recon = model(distorted_tensor).cpu().numpy()[0, 0]

            # calculate errors
            normal_error = np.mean((original - original_recon) ** 2)
            anomaly_error = np.mean((distorted - distorted_recon) ** 2)

            axes[0, i].imshow(original, cmap='gray')
            axes[0, i].set_title(f'Original\nMSE: {normal_error:.4f}')
            axes[0, i].axis('off')

            axes[1, i].imshow(distorted, cmap='gray')
            axes[1, i].set_title(f'Distorted\nMSE: {anomaly_error:.4f}')
            axes[1, i].axis('off')

            axes[2, i].imshow(np.abs(distorted - distorted_recon), cmap='hot')
            axes[2, i].set_title(f'Error Map\nRatio: {anomaly_error / normal_error:.2f}x')
            axes[2, i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_statistical_summary(results):
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION STATISTICAL ANALYSIS SUMMARY")
    print("=" * 80)

    normal_errors = np.array(results['normal_errors'])
    print(f"\nNormal Images Statistics:")
    print(f"  Mean MSE: {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f}")
    print(f"  Min/Max MSE: {np.min(normal_errors):.6f} / {np.max(normal_errors):.6f}")
    print(f"  Number of samples: {len(normal_errors)}")

    print(f"\nDistortion Analysis:")
    print("-" * 60)

    for distortion_type in results['statistical_tests']:
        print(f"\n{distortion_type.replace('_', ' ').title()} Distortion:")

        for intensity in results['statistical_tests'][distortion_type]:
            stats_data = results['statistical_tests'][distortion_type][intensity]
            roc_data = results['roc_metrics'][distortion_type][intensity]

            print(f"  Intensity {intensity}:")
            print(f"    Anomaly MSE: {stats_data['anomaly_mean']:.6f} ± {stats_data['anomaly_std']:.6f}")
            print(f"    Mean Increase: {(stats_data['anomaly_mean'] / stats_data['normal_mean'] - 1) * 100:.1f}%")
            print(f"    Cohen's d: {stats_data['cohens_d']:.3f}")
            print(f"    T-test p-value: {stats_data['t_pvalue']:.2e}")
            print(f"    Mann-Whitney p-value: {stats_data['mannwhitney_pvalue']:.2e}")

            if roc_data:
                print(f"    ROC AUC: {roc_data['auc_score']:.3f}")
                print(f"    Optimal Threshold: {roc_data['optimal_threshold']:.6f}")

            # Interpretation
            if stats_data['t_pvalue'] < 0.001:
                significance = "Highly Significant (***)"
            elif stats_data['t_pvalue'] < 0.01:
                significance = "Very Significant (**)"
            elif stats_data['t_pvalue'] < 0.05:
                significance = "Significant (*)"
            else:
                significance = "Not Significant"

            effect_magnitude = abs(stats_data['cohens_d'])
            if effect_magnitude < 0.2:
                effect_desc = "Small"
            elif effect_magnitude < 0.5:
                effect_desc = "Medium"
            elif effect_magnitude < 0.8:
                effect_desc = "Large"
            else:
                effect_desc = "Very Large"

            print(f"    Statistical Significance: {significance}")
            print(f"    Effect Size: {effect_desc}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Test Texture Autoencoder for Anomaly Detection')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='texture_patches',
                        help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples for anomaly testing')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')

    args = parser.parse_args()

    # output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    print("Loading trained model...")
    model, checkpoint = load_model(args.model_path, device)
    print(f"Model loaded from epoch {checkpoint['epoch']}")

    # load test data
    print("Loading test data...")
    train_paths, test_paths = load_texture_tiles(args.data_dir)

    if not test_paths:
        print("No test data found. Using training data for testing.")
        test_paths = train_paths

    print(f"Found {len(test_paths)} test images")

    test_dataset = TextureDataset(test_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # evaluation
    print("\nPerforming basic model evaluation...")
    basic_results = evaluate_model(model, test_loader, device)
    print(f"Average test loss: {basic_results['avg_loss']:.6f}")
    print("\nPerforming comprehensive anomaly detection evaluation...")
    anomaly_results = evaluate_anomaly_detection(
        model, test_paths, device,
        distortion_types=['sun_flare', 'grid_shuffle', 'spatter'],
        intensities=[0.5, 1.0, 1.5],
        num_samples=args.num_samples
    )

    print_statistical_summary(anomaly_results)

    # visualizations
    print("\nGenerating analysis plots...")
    plot_anomaly_analysis(anomaly_results, os.path.join(args.output_dir, 'plots'))
    print("Creating reconstruction visualizations...")
    visualize_reconstructions(
        model, test_paths, device, num_samples=6,
        save_path=os.path.join(args.output_dir, 'sample_reconstructions.png')
    )

    print(f"\nEvaluation completed! Results saved to {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
