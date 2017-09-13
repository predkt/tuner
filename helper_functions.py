import smtplib
import pandas as pd
import math
import numpy as np
import operator

import matplotlib.pyplot as plt

import time
import datetime
import os
import sys
import threading
from functools import wraps

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, roc_auc_score,log_loss
from sklearn.model_selection import GridSearchCV

from IPython.core.debugger import Tracer


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage

from os.path import basename


import cProfile
import pstats
from io import StringIO
import marshal
import tempfile
import pprint
import psutil
import re

import objgraph

import pickle
from os.path import exists


# bit-serialize any object.Creates a new file if none, else appends.
# returns pickled object if only path is supplied
# stores the object in a dictionary with obj_key as key
def pickler(path, obj_to_pickle = None, obj_key = None):


    save ={}
    
    if exists(path):
        try:
          f = open(path, 'rb')
          save = pickle.load(f)
          f.close()
        except Exception as e:
          print('Unable to read data from', context['pickle'], ':', e)
          raise

    if(obj_to_pickle):
        save.update({obj_key: obj_to_pickle})

        try:
          f = open(path, 'wb')
          pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
          f.close()
        except Exception as e:
          print('Unable to save data to', context['pickle'], ':', e)
          raise
    
    return save




# Writes a dictionary to the file at supplied path.
# Optional description text describing the dictionary

def write_dict(d, path, description =''):

    
    with open(path, "a") as f:
      h_line = '-------------------------------\n'
      f.write(h_line)
      f.write(description +'\n')
      for k, v in d.items():
        if isinstance(v, dict):
          write_dict(v, path)
        else:
            tmp_str = str(k) + ' : ' + str(v) +'\n'
            f.write(tmp_str)
      f.write(h_line)
    
    return(d)


#computes max_memory and cpu usage from dictionary of measured results 
def max_stats(profile_results, context):
    cpu_list= []
    used_memory_list =[]
    active_memory_list =[]
    total_memory_list = []
    buffered_memory_list =[]
    cached_memory_list =[]
    shared_memory_list = []
    swap_memory_list = []
    return_dict= {}


    for i, (key,value) in enumerate(profile_results.items()):

        if not key == 'max_memory':
            cpu_list.append(value['all_cpu'])             
            total_memory_list.append(value['memory'][0])
            used_memory_list.append(value['memory'][3])
            active_memory_list.append(value['memory'][5])
            buffered_memory_list.append(value['memory'][7])
            cached_memory_list.append(value['memory'][8])
            shared_memory_list.append(value['memory'][9])
            swap_memory_list.append(value['swap'][0])
            
            
    max_memory = profile_results['max_memory']  
    
    return_dict.update({'max_cpu': np.max(cpu_list)})
    return_dict.update({'total_memory': convert_size(np.max(total_memory_list))})
    return_dict.update({'max_used_memory': convert_size(np.max(used_memory_list))})
    return_dict.update({'max_active_memory': convert_size(np.max(active_memory_list))})
    return_dict.update({'max_buffered_memory': convert_size(np.max(buffered_memory_list))})
    return_dict.update({'max_cached_memory': convert_size(np.max(cached_memory_list))})
    return_dict.update({'max_shared_memory': convert_size(np.max(shared_memory_list))})
    return_dict.update({'max_swapped_memory': convert_size(np.max(swap_memory_list))})
    return_dict.update({'max_thread_memory': max_memory})

    
    write_dict(return_dict, context['summary'], 'Maximum Usage Stats')
    pickled = pickler(context['pickle'], return_dict, 'max stats')
    
    return return_dict



#sends email to self from self, with passed subject and body
#Files to attach can be passed as list to the 'files' argument

def send_email(subject, body, version_list_html='', files=None, context = None):
    
    def prompt(prompt):
        return raw_input(prompt).strip()

    fromaddr = 'abhijeet.jha@gmail.com'
    toaddr  = 'abhijeet.jha@gmail.com'
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = subject
    
    body = body
    
    msg.attach(MIMEText(body, 'html'))
    
    
    footer ="<br>><hr>" + version_list_html
    msg.attach(MIMEText(footer, 'html'))

    
    #######################################
#     To embed accuracy image
#     pickled = pickler(context['pickle'])
#     img = pickled['accuracy plot']



#     # This example assumes the image is in the current directory
#     fp = open(img, 'rb')
#     msgImage = MIMEImage(fp.read())
#     fp.close()

#     # Define the image's ID as referenced above
#     msgImage.add_header('Content-ID', '<image1>')
#     msg.attach(msgImage)


####################################
    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)

    
 
    smtp_server = 'email-smtp.us-east-1.amazonaws.com'
    smtp_username = 'AKIAJFYKGSZH6TNFD2WQ'
    smtp_password = 'AoSGycN2iVoV9b/eDhm6ht2ZK7OaRa58InGKywLQ/nfF'
    smtp_port = '587'
    smtp_do_tls = True

    server = smtplib.SMTP(
        host = smtp_server,
        port = smtp_port,
        timeout = 10
        )
    server.starttls()
    server.ehlo()
    server.login(smtp_username, smtp_password)
    
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)

    
# create html markup for a dictionay. 
# Note - doesnt work with nested dictionaries
#TO DO - write an iterator 
def dict_to_html(dict):
    df=pd.DataFrame(dict)
    outhtml= df.to_html(na_rep = "", index = True).replace('border="1"','border="0"')
    outhtml=outhtml.replace('<th>','<th style = "display: none">')
    outhtml=outhtml.replace('<td>','<td style= "padding: 8px;text-align: left;border-bottom: 1px solid #ddd;;">')
    outhtml=outhtml.replace('table','table width = "100%"')
    return outhtml


def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

# dataset and labels are of type np.ndarray, returned by merge_dataset()
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


def fetch_paths():
    today = datetime.date.today().strftime("%Y%m%d")
    now = time.strftime("%H%M%S",time.gmtime())

    
    model_path = 'savedmodels/' + today +'/'
    log_path = 'logs/' + today +'/'
    stats_path = 'stats/' + today +'/'
    runprofiles_path = 'runprofiles/' + today +'/'
    pickle_path = 'runpickles/' + today + '/'
    plot_path = 'plots/' + today + '/' + 'run_' + now +'/'
    
    current = str(os.getcwd())
    log_root = os.path.join(log_path)
    model_root = os.path.join(model_path)
    stats_root = os.path.join(stats_path)
    runprofiles_root = os.path.join(runprofiles_path)
    pickled_root = os.path.join(pickle_path)
    plot_root = os.path.join(plot_path)

    if not os.path.exists(model_root):
        os.makedirs(model_root)

    if not os.path.exists(log_root):
        os.makedirs(log_root)
        
    if not os.path.exists(stats_root):
        os.makedirs(stats_root)
        
    if not os.path.exists(runprofiles_root):
        os.makedirs(runprofiles_root)
    
    if not os.path.exists(pickled_root):
        os.makedirs(pickled_root)
    
    if not os.path.exists(plot_root):
        os.makedirs(plot_root)
        
    summary = runprofiles_root + 'summary_' + today + now + '.txt'
    pickle = pickled_root + 'run_' + today + now 
    modelpickles = model_root + 'pickled_' + today + now
    statsfile = stats_root + 'stats_' + today + now
    
    context ={}
    context.update({'log path': log_root})
    context.update({'plot path': plot_root})
    context.update({'model path': model_root})
    context.update({'stats path': stats_root})
    context.update({'runprofiles path': runprofiles_root})
    context.update({'run date': today})
    context.update({'run time': now})
    context.update({'summary': summary})
    context.update({'pickle': pickle})
    context.update({'modelpickles': modelpickles})
    context.update({'statsfile': statsfile})
    
    return context


def html_class_name(class_name):
    #class_name = class_name.replace("<class '", "")
    class_name = class_name.replace(">", "")
    class_name = class_name.replace("<", "")
    class_name = class_name.replace("'", "")
    class_name = class_name.replace(" ", "")
    class_name = class_name.replace(".", "")
    class_name = class_name.replace(":", "")
    return class_name



# Routine to add commas to a float string
def commify3(amount):
    amount = str(amount)
    amount = amount[::-1]
    amount = re.sub(r"(\d\d\d)(?=\d)(?!\d*\.)", r"\1,", amount)
    return amount[::-1]



def save_summary(context, stats_file_path):
    #print (" --------------------------------------------------------------------")
    #summary = context['runprofiles path'] + 'summary_'+ context['run time'] +'.txt'
    stream = open(os.path.join(context['summary']), 'a');
    stats = pstats.Stats(stats_file_path, stream=stream)
    pprint.pformat(stats.strip_dirs().sort_stats('cumtime').print_stats(15))
    stream.flush()
    stream.close()




def poll_system_profile(context, interval=0.0):
    #log_root, model_root, stats_root, today, now = fetch_paths()    
    num_cpu =psutil.cpu_count()
    percpu_list =[]
    
    # Current system-wide CPU utilization as a percentage
    # ---------------------------------------------------
 
    # Individual CPUs
    sys_percs_percpu = psutil.cpu_percent(interval, percpu=True)
    
    
    for cpu_num, perc in enumerate(sys_percs_percpu):
        percpu_list.append(perc)
    # end for
 
 
    # Details on Current system-wide CPU utilziation as a percentage
 
    # --------------------------------------------------------------
    # Server as a whole
    overall_cpu = np.mean(percpu_list)
    sys_percs_total_details = psutil.cpu_times_percent(interval, percpu=False)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    used = mem.total - mem.available
    sys_cpu_times = {}
    
    sys_cpu_times.update({'profile_time': datetime.date.today().strftime("%Y%m%d") + time.strftime("%H%M%S",time.gmtime()) })
    sys_cpu_times.update({'all_cpu': overall_cpu})
    sys_cpu_times.update({'per_cpu': sys_percs_percpu})
    sys_cpu_times.update({'memory': mem})
    sys_cpu_times.update({'swap':swap})
    
    write_dict(sys_cpu_times, context['summary'], 'Usage Logging')
    

    
    return sys_cpu_times

   

def measure_memory_usage(context, target_call, target_args, log_interval=30, log_filename=None, memory_usage_refresh=0.01):
    """
    measure the memory usage of a function call in python.\n
    Note: one may have to restart python to get accurate results.\n
    :param target_call: function to be tested\n
    :param target_args: arguments of the function in a tuple\n
    :param memory_usage_refresh: how frequent the memory is measured, default to 0.005 seconds\n
    :return: max memory usage in kB (on Linux/Ubuntu 14.04), may depend on OS
    """
  

    class StoppableThread(threading.Thread):
        def __init__(self, target, args):
            super(StoppableThread, self).__init__(target=target, args=args)
            self.daemon = True
            self.__monitor = threading.Event()
            self.__monitor.set()
            self.__has_shutdown = False

        def run(self):
            '''Overloads the threading.Thread.run'''
            # Call the User's Startup functions
            self.startup()

            # use the run method from Superclass threading.Thread
            super(StoppableThread, self).run()

            # Clean up
            self.cleanup()

            # Flag to the outside world that the thread has exited
            # AND that the cleanup is complete
            self.__has_shutdown = True

        def stop(self):
            self.__monitor.clear()

        def isRunning(self):
            return self.__monitor.isSet()

        def isShutdown(self):
            return self.__has_shutdown

        def mainloop(self):
            '''
            Expected to be overwritten in a subclass!!
            Note that Stoppable while(1) is handled in the built in "run".
            '''
            pass

        def startup(self):
            '''Expected to be overwritten in a subclass!!'''
            pass

        def cleanup(self):
            '''Expected to be overwritten in a subclass!!'''
            pass

    class MyLibrarySniffingClass(StoppableThread):
        def __init__(self, target, args):
            super(MyLibrarySniffingClass, self).__init__(target=target, args=args)
            self.target_function = target
            self.results = None

        def startup(self):
            # Overload the startup function
            print ("Calling the Target Library Function...")

        def cleanup(self):
            # Overload the cleanup function
            print ("Library Call Complete")

        #process = psutil.Process(os.getpid())

   
    process = psutil.Process(os.getpid())
    my_thread = MyLibrarySniffingClass(target_call, target_args)
    
    run_profile ={}
    start_mem = process.memory_full_info().uss  #uss
    
    sys_profile = poll_system_profile(context, interval=0.1)
    print ("Written to summary File")
    
    run_profile.update({time.strftime("%H:%M:%S",time.gmtime()): sys_profile})
    
    my_thread.start()
    delta_mem = 0
    max_memory = 0
    last_run=time.time()

    while(True):
        time.sleep(memory_usage_refresh)
        cur_time = time.time()
        del_time = cur_time - last_run
        
        
        
        if round(del_time) > log_interval:
            sys_profile = poll_system_profile(context)
            print ("Written to summary File")
            last_run = cur_time
            run_profile.update({time.strftime("%H:%M:%S",time.gmtime()): sys_profile})
            #print(run_profile)
        
        current_mem = process.memory_info().rss 
        delta_mem = current_mem - start_mem
        if delta_mem > max_memory:
            max_memory = delta_mem

            
        if my_thread.isShutdown():
            print ("Memory measurement complete!")
            break

    current_mem = process.memory_full_info().uss  #uss
    delta_mem = current_mem - start_mem
    if delta_mem > max_memory:
        max_memory = delta_mem



    print ("MAX Memory Usage in MB: {}".format( convert_size(max_memory)))

    
    run_profile.update({time.strftime("%H:%M:%S",time.gmtime()): sys_profile})
    run_profile.update({'max_memory': convert_size(max_memory)})
   
    
    written = max_stats(run_profile, context)
    
    return written



def objects_growth(path, description = ''):
    
    
    orig_stdout = sys.stdout
    
    f = open(path, 'a')
    sys.stdout = f
    
    print(description)
    f.flush()
    
    print(sys.version)
    print("---------------")
    print("Object Growth")
    print(objgraph.show_growth()) 
    f.flush()
    
    f.close()
    
    sys.stdout = orig_stdout
    #return''

def modelfit(alg, datasets, labels, context, metrics, useTrainCV=True, cv_folds=3, early_stopping_rounds=20, num_labels = None):
      
    try:
          train_dataset= datasets[0]
          train_labels = labels[0]
    except Exception as e:
          print('Unable to save data to load training samples', e)
          raise
    
    valid_dataset = datasets[1]
    test_dataset = datasets[2]
    
    #train_labels = labels[0]
    valid_labels = labels[1]
    test_labels = labels[2]

    
    run_stats={}
    optimal_boosters = 0
    num_class = num_labels
    
    
    if useTrainCV:

        
        xgb_param = alg.get_xgb_params()
        xgb_param.update({'num_class': num_class})
        run_stats.update({'original parameters': xgb_param})


        xgtrain = xgb.DMatrix(train_dataset,label=train_labels)

        
        cv_start_time = time.time()
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,metrics=metrics, early_stopping_rounds=early_stopping_rounds)
        cv_end_time = time.time()
        

        
        cv_time_raw = cv_end_time - cv_start_time
        cv_time = time.strftime("%H:%M:%S s",time.gmtime(cv_time_raw))
        run_stats.update({'cv run time': cv_time})

        
 
        alg.set_params(n_estimators = cvresult.shape[0])
        optimal_boosters = cvresult.shape[0]
        run_stats.update({'optimal_boosters': optimal_boosters})
        
    #Fit the algorithm on the data
    fit_start_time =time.time()
    alg.fit(train_dataset, train_labels,eval_metric=metrics)
    fit_end_time =time.time()
    
    fit_time_raw = fit_end_time - fit_start_time
    fit_time = time.strftime("%H:%M:%S s",time.gmtime(fit_time_raw))
    run_stats.update({'fit time': fit_time})
    #print(run_stats)
        
    #Predict training and validation set:
    predict_start_time = time.time()
    dtrain_predictions = alg.predict(train_dataset)

    dvalid_predictions = alg.predict(valid_dataset)
    dtest_predictions = alg.predict(test_dataset)

    predict_end_time = time.time()
    
    predict_time_raw = predict_end_time - predict_start_time
    predict_time = time.strftime("%H:%M:%S s",time.gmtime(predict_time_raw))
    run_stats.update({'predict time': predict_time})
    #print(run_stats)
        
     #Print model report:
    acc_score_train = accuracy_score(train_labels, dtrain_predictions)

    acc_score_valid = accuracy_score(valid_labels, dvalid_predictions)
    acc_score_test = accuracy_score(test_labels, dtest_predictions)

    print ("\nModel Report")
    print ("Accuracy : {0:.5f}".format(acc_score_train)) 
    print ("Optimal Boosters : {}".format(optimal_boosters)) 
    
    run_stats.update({' Train Accuracy': acc_score_train})
    if acc_score_valid: run_stats.update({' Validation Accuracy': acc_score_valid})
    if acc_score_test: run_stats.update({' Test Accuracy': acc_score_test})

    booster = alg.booster()
    fit_parameters = booster.attributes()
    run_stats.update({'fit attributes': fit_parameters})
    class_name = html_class_name(str(booster.__class__))
    
    #print(now)
    fname = context['model path'] + str(class_name) + context['run date'] + context['run time']
    
    
    alg.booster().save_model(fname)
    run_stats.update({'saved model path': fname})
    pickled = pickler(context['modelpickles'], alg, 'model')
    run_stats.update({'pickled model': context['modelpickles']})
    
    feat_imp_ser = pd.Series(alg.booster().get_fscore()).head(10).sort_values(ascending=False)
    feat_dict = feat_imp_ser.to_dict()
    run_stats.update({'Feature Importance Score': feat_dict})
    #print(run_stats)
     
    write_dict(run_stats, context['summary'], '#Booster Optimize Run')
    pickled = pickler(context['pickle'], run_stats, 'model results')
    
    #plotCV(cvresult, acc_score_train, acc_score_valid)  

    plotCV(cvresult, optimal_boosters, context, acc_score_train, acc_score_valid, acc_score_test)

    
    
    ##########Book keeping - update optimal parameters in dictionary with new boosters
    
    parameters = xgb_param

    #native xgboost requires num_class, scikit_learn doesnt like it
    del parameters['num_class']

    #update with results
    parameters.update({'n_estimators': optimal_boosters})

    updated_pickle =pickler(context['pickle'], parameters, 'optimal parameters')
    
    return updated_pickle
    ########## End Book Keeping
 


def plotCV(cvresult, optimal_boosters, context, accuracy_train = 0, accuracy_valid = 0, accuracy_test = 0,  title ='Accuracy Score by Tree Growth', ylim=(0.7,1)):

    # ylim=(0.8,1.01)
    
    plt.rcParams['figure.figsize'] = (20,10)
    cvresult_df = pd.DataFrame(cvresult)
    x_values = list(range(cvresult_df.shape[0]))
    test_error = cvresult_df.iloc[:,0].tolist()
    test_std = cvresult_df.iloc[:,1].tolist()
    
    train_error = cvresult_df.iloc[:,2].tolist()
    train_std = cvresult_df.iloc[:,3].tolist()
    
    x_values_int= [None]*len(x_values)
    test_error_float= [None]*len(x_values)
    test_std_float= [None]*len(x_values)
    train_error_float= [None]*len(x_values)
    train_std_float= [None]*len(x_values)
    
    for i in range(len(x_values)):
        x_values_int[i] = int(x_values[i])
        test_error_float[i] = 1 - float(test_error[i])
        test_std_float[i] = float(test_std[i])
        train_error_float[i] = 1 - float(train_error[i])
        train_std_float[i] = float(train_std[i])
        
      
    fig = plt.figure()
    plt.xlabel('Number of Boosters')
    plt.ylabel('Accuracy')
    #plt.ylim(0.7,1,1)

    plt.plot(x_values_int,
         train_error_float,
         label='Training Score',
         color = 'r')

    plt.plot(x_values_int,
         test_error_float,
         label='CV Score',
         color = 'g')


    plt.fill_between(x_values_int,
                np.array(train_error_float) - np.array(train_std_float),
                np.array(train_error_float) + np.array(train_std_float),
                alpha =0.2, color ='r')

    plt.fill_between(x_values_int,
                np.array(test_error_float) - np.array(test_std_float),
                np.array(test_error_float) + np.array(test_std_float),
                alpha =0.2, color ='g')


    plt.axhline(y = 1, color='k', ls ='dashed')
    plt.axvline(x = optimal_boosters, ls ='dashed', label ='Optimal #Boosters')

    plt.plot(optimal_boosters, float(accuracy_train), marker='o', markersize=6, color="blue", label = 'Train Accuracy')
    plt.plot(optimal_boosters, float(accuracy_valid), marker='o', markersize=6, color="maroon", label = 'Valid Accuracy')

    plt.plot(optimal_boosters, float(accuracy_test), marker='3', markersize=6, color="green", label = 'Test Accuracy')

    
    plt.text(optimal_boosters+5, float(accuracy_train), float(accuracy_train), fontsize =12, 
             bbox=dict(facecolor='none', edgecolor='blue', boxstyle='round,pad=1'))
    
    plt.text(optimal_boosters+5, float(accuracy_valid), float(accuracy_valid), fontsize =12, 
             bbox=dict(facecolor='none', edgecolor='maroon', boxstyle='round,pad=1'))
    

    plt.text(optimal_boosters+5, float(accuracy_test), float(accuracy_test), fontsize =12, 

             bbox=dict(facecolor='none', edgecolor='green', boxstyle='round,pad=1'))
    
    
    plt.legend(loc = 'best')
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
    now = time.strftime("%H%M%S",time.gmtime())
    save_file = context['plot path'] + now + '.png'
    fig.savefig(save_file, bbox_inches='tight')
    pickled = pickler(context['pickle'], save_file, 'accuracy plot')


def extend_single_param(result, delta_step, allowed_range, seen):
    
    new_list ={'left': round(result - delta_step, 2), 'right': round(result + delta_step, 2)}
    
    
    if new_list['left'] <= allowed_range[0] or new_list['left'] in seen:
        del new_list['left']
    
    if new_list['right'] > allowed_range[1] or new_list['right'] in seen:
        del new_list['right']
    
    return list(new_list.values())



def extend_param_dict(current_best, steps, allowed_ranges, seen):
    #first find the extended range for each parameter
    # step and allowed range and dictionaries for values for each tunable parameters
    parameters ={}
    for k, v in current_best.items():
        seen_list = seen[k]
        step =steps[k]
        allowed_range = allowed_ranges[k]
        print('parameter', k)
        print('result:',v,' step:', step,' allowed_range:', allowed_range,' seen:', seen_list)
        new_range = extend_single_param( v, step, allowed_range, seen_list)
        print(new_range)
        parameters.update({k:new_range})
    
    #iterate through new parameters - if none, set to incoming current best value
    
    for k, v in parameters.items():
        if not v: parameters[k] = [current_best[k]]
        
    return parameters


def remove_duplicates(inlist):
    outlist =[]
    for i in inlist:
        if i not in outlist:
            outlist.append(float(i))
    
    return outlist
    

def tuner_cv(train_set, train_labels, val_set, val_labels, param_test, tuning_rounds, steps, allowed_ranges, context, scoring ='accuracy', cv = 3, val_tuned =True):
    pickled = pickler(context['pickle'])
    parameters = pickled['optimal parameters']
    
    tuning_results_params ={}
    tuning_results_accuracy ={}
    tuning_validation_accuracy ={}
    rounds_to_tune = tuning_rounds

    current_tuning_round = 0
    estimator = XGBClassifier(**parameters)

    tuned = False
    param_test = param_test
    seen = param_test


    while not tuned:
    
        loop_result =()
        
        #update seen with parameters already tested
        seen = { k:  seen[k] + param_test[k]  for k in seen }
        seen = { k:  remove_duplicates(seen[k]) for k in seen }
        
        # Remove the duplicates
        #seen = list(set(seen))
    
        gsearch = GridSearchCV(estimator = estimator, 
                        param_grid = param_test, 
                        scoring= scoring,
                        n_jobs= -1,
                        cv= cv)

        loop_result = gsearch.fit(train_set, train_labels)
        
        # score on the validation dataset
        loop_result_val = loop_result.score(val_set, val_labels)

        tuning_results_params.update({'iter'+str(current_tuning_round): loop_result.best_params_ })
        tuning_results_accuracy.update({'iter'+str(current_tuning_round): loop_result.best_score_ })
        tuning_validation_accuracy.update({'iter'+str(current_tuning_round): loop_result_val })
    
        print('Current Iteration ', loop_result.best_params_ , ' CV Accuracy ', loop_result.best_score_, ' Validation Accuracy ', loop_result_val)
    
    
        current_tuning_round = current_tuning_round + 1
    
        param_test = extend_param_dict(loop_result.best_params_, steps, allowed_ranges, seen)
        print("Extended List :", param_test)
        print('-------------------------------')
    
        #convert result dict values into list for comparison
        best_params_list ={k: [loop_result.best_params_ [k]] for k in loop_result.best_params_ }
        if param_test == best_params_list : tuned = True
        if current_tuning_round == rounds_to_tune:  tuned = True
    

    
    ##END WHILE TUNED  
    write_dict(seen, context['summary'],'Tested Values')

    #prepare dict for writing to file
    tuner_results_summary ={key: str(tuning_results_params[key]) + '  CV Accuracy: ' + str(tuning_results_accuracy[key]) + '  Validation Accuracy: ' + str(tuning_validation_accuracy[key]) for key in tuning_results_params.keys() }

    #compute the highest  CV accuracy 
    max_accuracy_key =max(tuning_results_accuracy, key=lambda key: tuning_results_accuracy[key])  
    
    #compute the highest Validation accuracy 
    if val_tuned: max_accuracy_key =max(tuning_validation_accuracy, key=lambda key: tuning_validation_accuracy[key])  

    
    # use CV accuracy for tuning
    #tuning_results_params[max_accuracy_key]
    
    # use Validation accuracy for traiing
    tuning_results_params[max_accuracy_key]

    #pprint.pprint(tuner_results_summary)
    write_dict(tuner_results_summary, context['summary'], 'Tuning Iterations')
    write_dict({'Chosen:': str(tuning_results_params[max_accuracy_key]) + ' CV Accuracy: ' + str(tuning_results_accuracy[max_accuracy_key]) + ' Validation Accuracy: ' + str(tuning_validation_accuracy[max_accuracy_key])}, context['summary'])
    

    # Get the optimal parameters from the run
    
    # use Validation accuracy
    params_to_update = tuning_results_params[max_accuracy_key]

    # Update the parameters list with the new updated values for the params tested
    parameters.update({k: params_to_update[k] for k in params_to_update.keys()})

    # Update the pickle
    updated_pickle = pickler(context['pickle'], parameters, 'optimal parameters')
    
    




    
    

