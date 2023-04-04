import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict


def others_train(version, model, p_model, Extra_modules, paggregation_models, train_loader, test_loader, optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, valid_value, client_idx, device, args, a_iter, phase):

    return 0, 0