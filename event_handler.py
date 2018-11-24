#! /usr/bin/python2
import logger, csv, torch
from pdb import set_trace as bp
from proxy_configuration import *

class Metric(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.counter = 0
        self.difference = 0
        self.value = 0

    def increment(self, value = None):
        if value:
            self.difference += value - self.value
            self.value = value

        self.counter += 1
        if self.counter % self.threshold == 0:
            return True
        return False

    def get_value(self):
        tmp = self.difference
        self.difference = 0
        return tmp

    def get_index(self):
        return self.counter / self.threshold


class EventHandler(object):
    """
    Handles all the logging and model validation related events.

    Parameters:
    ___________
        session_name : str
            name by which you can find a particular session
    """

    visdom_opts = {"server":'http://localhost', "port":8097}
    def __init__(self, session_name ):
        stats = logger.Experiment(session_name ,
                log_git_hash =False, use_visdom=True, visdom_opts = self.visdom_opts,
                time_indexing = False, xlabel='reps')

        self.heatmap_windows = {}
        self.visdom = stats.plotter.viz
        self.metrics = {
            "avg_steps":Metric(LOGGING_AVERAGE),
            "n_explorations":Metric(LOGGING_AVERAGE),
            "n_exploitations":Metric(LOGGING_AVERAGE),
            "avg_reward":Metric(LOGGING_AVERAGE),
            "model_computation_time":Metric(BATCH_SIZE),
            "loss_backward_time":Metric(BATCH_SIZE)
            }
        self.train_metrics = stats.ParentWrapper(tag="training", name="toy",
                                            children=(
                                                stats.AvgMetric(name='avg_steps'),
                                                stats.BestMetric(name="best_n_steps", mode="min"),

                                                stats.AvgMetric(name="avg_reward"),
                                                stats.SimpleMetric(name="avg_memory_reward"),

                                                stats.SimpleMetric(name="loss"),
                                                stats.SimpleMetric(name="norm"),
                                                stats.SimpleMetric(name="optimizer_avg_predicted_next_state_value"),
                                                stats.SimpleMetric(name="optimizer_avg_main_model_state_action_value"),

                                                stats.SimpleMetric(name="norm_delta_embedding"),
                                                stats.SimpleMetric(name="n_explorations"),
                                                stats.SimpleMetric(name="n_exploitations"),

                                                stats.SimpleMetric(name = "validation_reward_per_problem"),
                                                stats.SimpleMetric(name = "validation_steps_per_problem"),
                                                stats.SimpleMetric(name = "validation_avg_discounted_action_value"),

                                                # profiler metrics
                                                stats.SimpleMetric(name = "self_rss"),
                                                stats.SimpleMetric(name = "child_rss"),
                                                stats.AvgMetric(name="model_computation_time"),
                                                stats.AvgMetric(name="loss_backward_time"),
                                                stats.SimpleMetric(name="optimizer_step_time"),
                                                stats.SimpleMetric(name="optimizer_sampling_time"),
                                                stats.SimpleMetric(name="validation_time"),
                                                stats.SimpleMetric(name="total_optimization_time"),
                                    ))
        self.train_metrics.reset()


    def plot_heatmap(self, data, counter_name, counter, title):
        if title not in self.heatmap_windows:
            self.heatmap_windows[title] = self.visdom.heatmap(torch.randn(10, 1), opts= {'title':title})

        self.visdom.heatmap(data,
                    opts= {'title':"{} ({} = {})".format(title, counter_name, counter)},
                    win = self.heatmap_windows[title]
                    )

    def log_metrics(self, metrics, index):
        """
        Logs numeric metric using logger.Experiment.

        Parameters:
        ______________
            metrics: list(tuple(str, value))
                (name_of_the_metric, value)

            index: int
                x coordinate
        """
        assert type(metrics) == list, "improper input..."
        for key,value in metrics:
            self.train_metrics.children[key].update(value).log(index)

    def update_metrics(self, metrics, difference = False):
        """
        mainly used for
            train_metrics.children for which avgMetric is available(difference = False)
            train_metrics.children for which difference between the last updated  value and current value needs to be plotted (difference = True)

        Parameters:
        ______________
            metrics: list(tuple(str, value))
                (name_of_the_metric, value)

            difference: bool
                if True, plot the difference from the last updated value

        """
        if difference:
            for key,value in metrics:
                if self.metrics[key].increment(value):
                    self.log_metrics([(key,self.metrics[key].get_value())], index = self.metrics[key].get_index())
        else:
            for key,value in metrics:
                self.train_metrics.children[key].update(value)
                if self.metrics[key].increment():
                    self.train_metrics.children[key].log_and_reset(idx = self.metrics[key].get_index())

    def append_stats(self, metrics):
        for key, value in metrics:
            self.stats[key].append(value)

    def make_a_text_window(self, string):
        self.visdom.text(string)

    def write_error(self, filename, string):
        with open(filename,'a') as f:
            f.write(string)
