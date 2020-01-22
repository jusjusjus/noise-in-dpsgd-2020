
from collections import namedtuple
from tensorflow_privacy.privacy.analysis.rdp_accountant import (
    compute_rdp, get_privacy_spent)

SpentDP = namedtuple('SpentDP', 'eps delta')

def compute_renyi_privacy(num_examples, batch_size, steps, sigma, delta):
    """compute privacy loss using Renyi Differential-Privacy estimate"""

    sampling_ratio = batch_size / num_examples
    orders = [1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] \
        + list(range(5, 64)) + [128, 256, 512]

    rdp = compute_rdp(sampling_ratio, sigma, steps, orders)
    epsilon, _, alpha = get_privacy_spent(orders, rdp, target_delta=delta)

    return SpentDP(epsilon, delta)
