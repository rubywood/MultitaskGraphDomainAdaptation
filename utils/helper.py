from tqdm import tqdm
import logging

def create_pbar(subset_name: str, num_steps: int):
    """Create a nice progress bar."""
    pbar_format = (
        "Processing: |{bar}| {n_fmt}/{total_fmt}" "[{elapsed}<{remaining},{rate_fmt}]"
    )
    pbar = tqdm(total=num_steps, leave=True, bar_format=pbar_format, ascii=True)
    if subset_name == "train":
        pbar_format += "step={postfix[1][step]:0.5f}" "|EMA={postfix[1][EMA]:0.5f}"
        # * Changing print char may break the bar so avoid it
        pbar = tqdm(
            total=num_steps,
            leave=True,
            initial=0,
            bar_format=pbar_format,
            ascii=True,
            postfix=["", dict(step=float("NaN"), EMA=float("NaN"))],
        )
    return pbar


class ScalarMovingAverage(object):
    """Class to calculate running average."""

    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def __call__(self, step_output):
        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # Calculate the exponential moving average
                new_ema_value = (
                    old_ema_value * self.alpha + (1.0 - self.alpha) * current_value
                )
                self.tracking_dict[key] = new_ema_value
            else:  # Init for variable which appear for the first time
                new_ema_value = current_value
                self.tracking_dict[key] = new_ema_value


def reset_logging(save_path):
    """Reset logger handler."""
    log_formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    log = logging.getLogger()  # Root logger
    for hdlr in log.handlers[:]:  # Remove all old handlers
        log.removeHandler(hdlr)
    new_hdlr_list = [
        logging.FileHandler(f"{save_path}/debug.log"),
        logging.StreamHandler(),
    ]
    for hdlr in new_hdlr_list:
        hdlr.setFormatter(log_formatter)
        log.addHandler(hdlr)