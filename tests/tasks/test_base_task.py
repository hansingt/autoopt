#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import os
import pickle

from autoopt.tasks import BaseTask


def create_bas_task(input_path=None, optimizer=None):
    input_path = input_path if input_path is not None else os.path.realpath(".")
    optimizer = optimizer if optimizer is not None else "Dummy"
    return BaseTask(input_path, optimizer)


def test_BaseTask_input_path():
    input_path = os.path.realpath(".")
    task = create_bas_task(input_path=input_path)
    assert task.input_path == input_path


def test_BaseTask_optimizer():
    optimizer = "Dummy"
    task = create_bas_task(optimizer=optimizer)
    assert task.optimizer == optimizer


def test_pickle_BaseTask():
    task = create_bas_task()
    task_pickle = pickle.dumps(task)
    check_task = pickle.loads(task_pickle)
    assert task.input_path == check_task.input_path
    assert task.optimizer == check_task.optimizer
