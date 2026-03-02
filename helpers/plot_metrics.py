import os
import sys
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics from TensorBoard logs.")
    parser.add_argument("run_dir", type=str, help="Path to the run directory containing TensorBoard logs.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the plots. Defaults to 'plots' inside run_dir.")
    return parser.parse_args()

def find_event_file(run_dir):
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if "events.out.tfevents" in file:
                return os.path.join(root, file)
    return None

def extract_scalar(ea, tag):
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return None, None

def plot_metric(steps, values, title, ylabel, output_path, color='blue'):
    if not steps or not values:
        print(f"No data for {title}, skipping...")
        return

    plt.figure(figsize=(10, 6))
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use('ggplot')
    
    plt.plot(steps, values, label=title, color=color, linewidth=2)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    args = parse_args()
    
    run_dir = args.run_dir
    event_file = find_event_file(run_dir)
    
    if not event_file:
        print(f"Error: No events.out.tfevents file found in {run_dir}")
        sys.exit(1)
        
    print(f"Reading events from: {event_file}")
    
    # Load the event file
    # Initialize with size_guidance to ensure we get all data
    # Use string keys to avoid attribute errors with different tensorboard versions
    ea = EventAccumulator(event_file,
        size_guidance={ 
            'scalars': 0,
        })
    ea.Reload()
    
    # Define output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(run_dir, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to plot
    metrics_to_plot = [
        {"tag": "train/loss", "title": "Training Loss", "ylabel": "Loss", "filename": "train_loss.pdf", "color": "#E24A33"}, # Red
        {"tag": "eval/loss", "title": "Validation Loss", "ylabel": "Loss", "filename": "val_loss.pdf", "color": "#348ABD"}, # Blue
        {"tag": "train/grad_norm", "title": "Gradient Norm", "ylabel": "Norm", "filename": "grad_norm.pdf", "color": "#988ED5"}, # Purple
        {"tag": "eval/mean_iou", "title": "Validation Mean IoU", "ylabel": "Mean IoU", "filename": "val_mean_iou.pdf", "color": "#8EBA42"}, # Green
        {"tag": "train/learning_rate", "title": "Learning Rate", "ylabel": "LR", "filename": "learning_rate.pdf", "color": "#FBC15E"}, # Yellow
        {"tag": "train/epoch", "title": "Epoch", "ylabel": "Epoch", "filename": "epoch.pdf", "color": "gray"},
        {"tag": "eval/accuracy_overall", "title": "Overall Accuracy", "ylabel": "Accuracy", "filename": "val_accuracy.pdf", "color": "cyan"}
    ]
    
    # Check if seaborn is available for nicer plots, otherwise fallback is default matplotlib
    if HAS_SEABORN:
        print("Using seaborn for styling.")
    else:
        print("Seaborn not found, using default matplotlib style.")

    for metric in metrics_to_plot:
        steps, values = extract_scalar(ea, metric["tag"])
        if steps:
            plot_metric(
                steps, 
                values, 
                metric["title"], 
                metric["ylabel"], 
                os.path.join(output_dir, metric["filename"]),
                color=metric["color"]
            )
        else:
            # Try to find a close match if exact tag not found?
            # For now just strictly check exact tags or basic variations?
            # Let's double check 'eval/loss' vs 'eval_loss' if needed, but based on previous steps we saw 'eval/loss'
            pass
            
    # Also try to combine train and val loss if possible (might be on different steps)
    train_steps, train_loss = extract_scalar(ea, "train/loss")
    val_steps, val_loss = extract_scalar(ea, "eval/loss")
    
    if train_steps and val_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_loss, label="Training Loss", color="#E24A33", alpha=0.6)
        plt.plot(val_steps, val_loss, label="Validation Loss", color="#348ABD", linewidth=2)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=0) # Loss is typically >= 0
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_val_loss_combined.pdf"), format='pdf', dpi=300)
        print(f"Saved combined loss plot to {os.path.join(output_dir, 'train_val_loss_combined.pdf')}")

if __name__ == "__main__":
    main()
