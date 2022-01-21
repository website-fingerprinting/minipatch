import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from train_model import train_model
from perturb_utils import perturb_trace, patch_length
from perturb_utils import verify_perturb
from process_utils import load_trained_model, load_data
from process_utils import load_checkpoint, save_checkpoint, del_checkpoint
from dual_annealing import dual_annealing

# Basic settings
training = False        # [True False]
model = 'DF'            # ['AWF' 'DF' 'VarCNN']
dataset = 'Sirinam'     # ['Sirinam' 'Rimmer100' 'Rimmer200' 'Rimmer500' 'Rimmer900']
num_sites = -1
num_samples = -1
verify_model = None     # [None 'AWF' 'DF' 'VarCNN']
verify_data = None      # [None '3d' '10d' '2w' '4w' '6w']
verbose = 1             # [0 1 2]

# Hyperparameters for patch generation
patches = 8             # [1 2 4 8]
inbound = 64            # [0 1 2 ... 64]
outbound = 64           # [0 1 2 ... 64]
adaptive = True         # [True False]
maxiter = 30            # [30 40 ... 100]
threshold = 1           # [0.9 0.91 ... 1.0]
polish = True           # [True False]

# Hyperparameters for Dual Annealing
initial_temp = 5230.
restart_temp_ratio = 2.e-5
visit = 2.62
accept = -1e3

class Minipatch:
    """
    Minipatch implementation.
    """
    def __init__(self, model, traces, labels, names=None, verbose=0, evaluate=True):
        self.model = model
        self.input_size = model.input_shape[1]
        self.num_classes = model.output_shape[1]

        self.traces = traces
        self.labels = labels.argmax(axis=-1)
        self.classes = names

        self.verbose = verbose

        # Evaluate to get samples with the correct prediction
        if evaluate:
            self.conf = model.predict(traces)
            self.preds = self.conf.argmax(axis=-1)
            self.correct = [np.where((self.preds == self.labels) &
                (site_id == self.labels))[0] for site_id in range(self.num_classes)]
    
    def perturb(self, num_sites, num_samples, bounds, adaptive, maxiter, threshold, polish, result_file):
        """
        Generate perturbation for each website and return all reseults.
        """
        # Get perturb traces for each website
        if num_sites == -1:
            site_traces = self.correct
        else:
            site_traces = self.correct[:num_sites]
        if num_samples != -1:
            site_traces = [ids[:num_samples] for ids in site_traces]
        
        # Load partial results if possible
        checkpoint, results = load_checkpoint(self, result_file)

        for site_id, trace_ids in enumerate(site_traces if self.verbose > 0 else tqdm(site_traces)):
            if site_id <= checkpoint:
                continue

            if self.verbose > 0:
                print('Perturbing website %s (%d/%d)...' % 
                    (str(site_id) if self.classes is None else self.classes[site_id],
                    site_id + 1, len(site_traces)),
                    end='\n' if adaptive or self.verbose > 1 else '\t')

            if len(trace_ids) == 0:
                if self.verbose > 0:
                    print('No valid traces')
                continue

            true_class = site_id

            if adaptive:
                # Perturb with adaptive perturbation bounds
                result = self.adaptive_tuning(site_id, trace_ids, true_class,
                            bounds, maxiter, threshold, polish)
            else:
                # Get website perturbations
                result = self.perturb_website(site_id, trace_ids, true_class,
                            bounds, maxiter, threshold, polish)
            
            if len(results) == 0:
                results = result.reset_index(drop=True)
            else:
                results = pd.concat([results, result], ignore_index=True)

            # Save partial results
            save_checkpoint(self, site_id, results, result_file)

        # Delete partial results
        del_checkpoint(result_file)

        # Save complete results
        results.to_json('%s.json' % result_file, orient='index')
    
    def adaptive_tuning(self, site_id, trace_ids, tar_class, bounds, maxiter, threshold, polish):
        """
        Find the best perturbation bounds for the website using binary search.
        """
        results = []
        trials = 0
        layer_nodes = [bounds]
        while len(layer_nodes) > 0:
            # Test each node in the layer
            for node in layer_nodes[::-1]:
                trials += 1
                if self.verbose > 0:
                    print('Trial %d - patches: %d - bounds: %d' % (
                        trials, node['patches'], max(node['inbound'], node['outbound'])),
                        end='\n' if self.verbose > 1 else '\t')

                # Get website perturbations
                result = self.perturb_website(site_id, trace_ids, tar_class,
                    node, maxiter, threshold, polish)
                
                # Remove unsuccessful node
                if result['successful'][0] == False:
                    layer_nodes.remove(node)
                
                # Record results whether successful or not
                if len(results) == 0:
                    results = result.reset_index(drop=True)
                else:
                    results = pd.concat([results, result], ignore_index=True)
            
            # Get the next layer of nodes
            children = []
            for node in layer_nodes:
                if node['patches'] > 1:
                    left_child = {
                        'patches': node['patches'] // 2,
                        'inbound': node['inbound'],
                        'outbound': node['outbound']}
                    if left_child not in children:
                        children.append(left_child)

                if max(node['inbound'], node['outbound']) > 1:
                    right_child = {
                        'patches': node['patches'],
                        'inbound': node['inbound'] // 2,
                        'outbound': node['outbound'] // 2}
                    if right_child not in children:
                        children.append(right_child)

            layer_nodes = children
        
        # Get the most successful result with the highest efficiency (num_success / patch_length)
        success = results[results['successful'] == True]
        if len(success) > 0:
            efficiency = success.apply(lambda x: x['num_success'] / patch_length(x['perturbation']), axis=1)
            best_idx = efficiency.iloc[::-1].idxmax()
        else:
            best_idx = results['num_success'].iloc[::-1].idxmax()

        return results.loc[best_idx:best_idx]
    
    def perturb_website(self, site_id, trace_ids, tar_class, bounds, maxiter, threshold, polish):
        """
        Generate perturbation for traces of a website.
        """
        # Define perturbation bounds for a flat vector of (y, β) values
        lengths = [len(np.argwhere(self.traces[i] != 0)) for i in trace_ids]
        length_bound = (1, np.percentile(lengths, 50))
        patches, inbound, outbound = bounds['patches'], bounds['inbound'], bounds['outbound']
        patch_bound = (-(inbound + 1), outbound + 1)
        perturb_bounds = [length_bound, patch_bound] * patches

        start = time.perf_counter()

        # Format the objective and callback functions for Dual Annealing
        def objective_func(perturbation):
            return self.predict_classes(self.traces[trace_ids], perturbation, tar_class)
        def callback_func(perturbation, f, context):
            return self.perturb_success(self.traces[trace_ids], perturbation, tar_class, threshold)

        # Call Scipy's implementation of Dual Annealing
        perturb_result = dual_annealing(
            objective_func, perturb_bounds,
            maxiter=maxiter,
            initial_temp=initial_temp,
            restart_temp_ratio=restart_temp_ratio,
            visit=visit,
            accept=accept,
            callback=callback_func,
            no_local_search=not polish,
            disp=True if self.verbose > 1 else False)

        end = time.perf_counter()

        # Record optimization results
        perturbation = perturb_result.x.astype(int).tolist()
        iteration = perturb_result.nit
        execution = perturb_result.nfev
        duration = end - start

        # Apply the optimized perturbation
        perturbed_traces = perturb_trace(self.traces[trace_ids], perturbation)
        # Note: model.predict() is much slower than model(training=False)
        predictions = self.model(perturbed_traces, training=False)
        predictions = np.array(predictions)

        # Calculate some statistics to return from this function
        true_class = site_id
        true_prior = self.conf[trace_ids, true_class]
        true_post = predictions[:, true_class]
        true_diff = true_prior - true_post

        pred_class = predictions.argmax(axis=-1)
        pred_prior = [conf[pred] for conf, pred in zip(self.conf[trace_ids], pred_class)]
        pred_post = [conf[pred] for conf, pred in zip(predictions, pred_class)]
        pred_diff = np.array(pred_post) - np.array(pred_prior)

        success = [pred != true_class for pred in pred_class]
        num_valid = len(trace_ids)
        num_success = sum(success)
        if num_success >= num_valid * threshold:
            successful = True
        else:
            successful = False
        
        # Result dictionary
        result = {'website': site_id, 'trace_ids': trace_ids, 'lengths': lengths,
            'num_valid': num_valid, 'num_success': num_success, 'successful': successful, 'success': success,
            'patches': patches, 'inbound': inbound, 'outbound': outbound, 'perturbation': perturbation,
            'iteration': iteration, 'execution': execution, 'duration': duration,
            'true_class': true_class, 'true_prior': true_prior, 'true_post': true_post, 'true_diff': true_diff,
            'pred_class': pred_class, 'pred_prior': pred_prior, 'pred_post': pred_post, 'pred_diff': pred_diff}

        if self.verbose > 0:
            print('%s - rate: %.2f%% (%d/%d) - iter: %d (%d) - time: %.2fs' % (
                'Succeeded' if num_success >= num_valid * threshold else 'Failed', 100 * num_success / num_valid,
                num_success, num_valid, iteration, execution, duration))

        return pd.DataFrame([result])

    def predict_classes(self, traces, perturbations, tar_class):
        """
        The objective function of the optimization problem.
        Perturb traces and get the model confidence.
        """
        perturbed_traces = perturb_trace(traces, perturbations)
        predictions = self.model(perturbed_traces, training=False)
        confidence = np.mean(np.array(predictions[:, tar_class]))

        # Minimize the function
        return confidence

    def perturb_success(self, traces, perturbation, tar_class, threshold):
        """
        The callback function of the optimization problem.
        Perturb traces and get the model predictions.
        """
        perturbed_traces = perturb_trace(traces, perturbation)
        predictions = self.model(perturbed_traces, training=False)
        pred_class = np.array(predictions).argmax(axis=-1)

        # Return True if the success rate is greater than the threshold
        num_success = sum([pred != tar_class for pred in pred_class])
        if num_success >= len(traces) * threshold:
            return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Minipatch: Undermining DNN-based Website Fingerprinting with Adversarial Patches')
    parser.add_argument('-t', '--train', action='store_true', default=training,
        help='Training DNN model for Deep Website Fingerprinting.')
    parser.add_argument('-m', '--model', default=model,
        help='Target DNN model. Supports ``AWF``, ``DF`` and ``VarCNN``.')
    parser.add_argument('-d', '--data', default=dataset,
        help='Website trace dataset. Supports ``Sirinam`` and ``Rimmer100/200/500/900``.')
    parser.add_argument('-nw', '--websites', type=int, default=num_sites,
        help='The number of websites to perturb. Take all websites if set to -1.')
    parser.add_argument('-ns', '--samples', type=int, default=num_samples,
        help='The number of trace samples to perturb. Take all samples if set to -1.')
    parser.add_argument('-vm', '--verify_model', default=verify_model,
        help='Validation Model. Default is the same as the target model.')
    parser.add_argument('-vd', '--verify_data', default=verify_data,
        help='Validation data. Default is the validation data. Supports ``3d/10d/2w/4w/6w`` with ``Rimmer200``.')
    parser.add_argument('--patches', type=int, default=patches,
        help='The number of perturbation patches.')
    parser.add_argument('--inbound', type=int, default=inbound,
        help='The maximum packet number in incoming patches. Perturb outgoing packets only if set to 0.')
    parser.add_argument('--outbound', type=int, default=outbound,
        help='The maximum packet number in outgoing patches. Perturb incoming packets only if set to 0.')
    parser.add_argument('--adaptive', action='store_true', default=adaptive,
        help='Adaptive tuning of patches and bounds for each website.')
    parser.add_argument('--maxiter', type=int, default=maxiter,
        help='The maximum number of iteration.')
    parser.add_argument('--threshold', type=float, default=threshold,
        help='The threshold to determine perturbation success.')
    parser.add_argument('--polish', action='store_true', default=polish,
        help='Perform local search at each iteration.')
    parser.add_argument('--verbose', type=int, default=verbose,
        help='Print out information. 0 = progress bar, 1 = one line per item, 2 = show perturb details.')

    # Parsing parameters
    args = parser.parse_args()
    training = args.train
    target_model = args.model
    dataset = args.data
    num_sites = args.websites
    num_samples = args.samples
    if args.verify_model is None:
        verify_model = target_model
    else:
        verify_model = args.verify_model
    if args.verify_data is None:
        verify_data = 'valid'
    else:
        verify_data = args.verify_data
    bounds = {
        'patches': args.patches,
        'inbound': args.inbound,
        'outbound': args.outbound}
    adaptive = args.adaptive
    optim_maxiter = args.maxiter
    success_thres = args.threshold
    optim_polish = args.polish
    verbose = args.verbose

    if training:
        train_model(target_model, dataset)
        os._exit(-1)

    result_dir = './results/%s_%s/' % (target_model.lower(), dataset.lower())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result_file = result_dir + '%s_%dpatches_%dinbound_%doutbound_%dmaxiter_%dthreshold%s_%swebsites_%ssamples' % (
        'adaptive' if adaptive else '', bounds['patches'], bounds['inbound'], bounds['outbound'],
        optim_maxiter, success_thres * 100, '_polish' if optim_polish else '',
        'all' if num_sites == -1 else str(num_sites), 'all' if num_samples == -1 else str(num_samples))
    
    if not os.path.exists('%s.json' % result_file):
        print('==> Loading %s model...' % target_model)
        model = load_trained_model(target_model, dataset)
        input_size = model.input_shape[1]
        num_classes = model.output_shape[1]
        data = 'test'

        print('==> Loading %s test data...' % dataset)
        traces, labels, names = load_data(dataset, input_size, num_classes, data)

        print('==> Start perturbing websites...')
        minipatch = Minipatch(model, traces, labels, names, verbose)
        minipatch.perturb(num_sites, num_samples, bounds, adaptive,
            optim_maxiter, success_thres, optim_polish, result_file)
        
    if verbose > 0:
        print('==> Loading %s model...' % verify_model)
    model = load_trained_model(verify_model, dataset, compile=True)
    input_size = model.input_shape[1]
    num_classes = model.output_shape[1]

    if verbose > 0:
        print('==> Loading %s %s data...' % (dataset, verify_data))
    traces, labels, names = load_data(dataset, input_size, num_classes, verify_data, verbose)

    if verbose > 0:
        print('==> Start verifying perturbation...')
    verify_perturb(model, traces, labels, verbose, result_file)
