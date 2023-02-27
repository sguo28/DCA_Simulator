import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np

root_path = f"logs/attack_test/"
writer = SummaryWriter(root_path)

for i in range(100):
    writer.add_scalar('system_metrics',
                      np.random.normal(10,5), i)
    time.sleep(1)
    writer.add_scalar('test2',1,i)