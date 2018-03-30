import smtplib
import pandas as pd
import math
import numpy as np
import operator
import string


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
import seaborn as sns

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



def get_new_context(version_list):

    
    context = fetch_paths()
    pickled = pickler(context['pickle'], context, 'run context')
    objects_growth(context['summary'], 'Beginning Heap')
    write_dict({'Installed Versions':version_list.__dict__['packages']}, context['summary'], 'Software Versions')
    write_dict(context, context['summary'], 'Run time Context')
    
    return context

def load_dataset(name, context):
    load_stats ={}
    size = 50
    image_size = 28
    num_labels =  10
    data = name

    data_pickle_path = '../../../../tensorflow/tensorflow/examples/udacity/' + data

    with open(data_pickle_path, 'rb') as f:
        data = pickle.load(f)

    train_dataset = data['train_dataset']
    length = train_dataset.shape[0]

    train_dataset = train_dataset.reshape(length, image_size*image_size)

    valid_dataset = data['valid_dataset']
    length = valid_dataset.shape[0]
    valid_dataset = valid_dataset.reshape(length, image_size*image_size)

    test_dataset = data['test_dataset']
    length = valid_dataset.shape[0]
    test_dataset = test_dataset.reshape(length, image_size*image_size)

    valid_labels = data['valid_labels']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

    #be nice to your RAM
    del data

    load_stats.update({'training dataset': train_dataset.shape})
    load_stats.update({'training labels': train_labels.shape})
    load_stats.update({'validations dataset': valid_dataset.shape})
    load_stats.update({'validation labels': valid_labels.shape})
    load_stats.update({'test dataset': test_dataset.shape})
    load_stats.update({'test labels': test_labels.shape})

    ############## WRITE TO SUMMARY FILE
    write_dict(load_stats, context['summary'],'Dataset Details')

    datasets = [train_dataset, valid_dataset, test_dataset]
    labels = [train_labels, valid_labels, test_labels]
    return datasets, labels



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
    smtp_username = 'A'
    smtp_password = 'A'
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


    
# parameter_ranges = {
#     'colsample_bylevel': [0.4, 1.0],
#     'colsample_bytree': [0.4, 1.0],
#     'subsample': [0.4, 1.0],

#     'learning_rate': [0, 1],
#     'n_estimators': [15, 1000],
    
#     'max_depth': [1,15],
#     'min_child_weight': [1,15],
#     'gamma': [0, 1],

#     'reg_alpha': [-3,2],   #powers of 10
#     'reg_lambda': [-3,2]}  #powers of 10



def tuner(diagnosis, in_parameters, parameter_ranges, context):
    out_parameters = in_parameters.copy()
    
    if diagnosis == 'High Variance':
    
        #'colsample_bylevel' - reduce value to increase regularization
        left = min(parameter_ranges['colsample_bylevel']) *100 
        right = in_parameters['colsample_bylevel'] *100 
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 
        
        out_parameters.update({'colsample_bylevel': new_value/100})
    
        #'colsample_bytree' - reduce value to increase regularization
        left = min(parameter_ranges['colsample_bytree']) *100 
        right = in_parameters['colsample_bytree'] *100 
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 
        
        out_parameters.update({'colsample_bytree': new_value/100})
    
        #'subsample' - reduce value to increase regularization
        left = min(parameter_ranges['subsample']) *100 
        right = in_parameters['subsample'] *100 
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 
        
        out_parameters.update({'subsample': new_value/100})
        
        #'max_depth' - reduce value to decrease model complexity
        left = min(parameter_ranges['max_depth']) 
        right = in_parameters['max_depth'] 
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 

        out_parameters.update({'max_depth': new_value})
        
        #'min_child_weight' - increase to reduce model complexity
        left = in_parameters['min_child_weight'] 
        right = max(parameter_ranges['min_child_weight']) 
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 
            
        out_parameters.update({'min_child_weight': new_value})
        
        #'gamma' - increase to reduce model complexity
        left = in_parameters['gamma']  * 100 
        right = max(parameter_ranges['gamma']) *100 
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right)  
        
        out_parameters.update({'gamma': new_value/100})
        
        #'alpha' - increase to reduce model complexity
        left =  (np.log10(1.0 / in_parameters['reg_alpha']) * -1) 
        right = max(parameter_ranges['reg_alpha']) 
        
        if left == right:
            new_value = np.power(10, float(left))
        else:
            new_value = np.power(10, float(np.random.randint(left,right)))
        
        
        new_value = round(new_value, 4)
        out_parameters.update({'reg_alpha':new_value})
        
        #'lambda' - increase to reduce model complexity
        left =  (np.log10(1.0 / in_parameters['reg_lambda']) * -1) 
        right = max(parameter_ranges['reg_lambda']) 
        
        if left == right:
            new_value = np.power(10, float(left))
        else:
            new_value = np.power(10, float(np.random.randint(left,right)))
        

        out_parameters.update({'reg_lambda': round(new_value,4)})
    
    
    if diagnosis == 'High Bias':
    
        #'colsample_bylevel' - increase value to reduce regularization
        left = in_parameters['colsample_bylevel'] *100 
        right = max(parameter_ranges['colsample_bylevel']) *100 +1
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 
        

        out_parameters.update({'colsample_bylevel': new_value/100})
    
        #'colsample_bytree' - increase value to reduce regularization
        left = in_parameters['colsample_bytree'] *100 
        right = max(parameter_ranges['colsample_bytree']) *100 +1
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 

        out_parameters.update({'colsample_bytree': new_value/100})
    
        #'subsample' - increase value to reduce regularization
        left = in_parameters['subsample'] *100 
        right = max(parameter_ranges['subsample']) *100 +1
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 
        
        out_parameters.update({'subsample': new_value/100})
        
        #'max_depth' - increase value to increase model complexity
        left = in_parameters['max_depth'] 
        right = max(parameter_ranges['max_depth']) +1
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 
        
        out_parameters.update({'max_depth': new_value})
        
        #'min_child_weight' - decrease to increase model complexity
        right = in_parameters['min_child_weight'] 
        left = min(parameter_ranges['min_child_weight'])
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 

        out_parameters.update({'min_child_weight': new_value})
        
        #'gamma' - reduce to increase model complexity
        right = in_parameters['gamma']  * 100 
        left = min(parameter_ranges['gamma']) *100 
        
        if left == right:
            new_value = left
        else:
            new_value = np.random.randint(left,right) 
        
        out_parameters.update({'gamma': new_value/100 })
        
        #'alpha' - decrease to increase model complexity
        right =  (np.log10(1.0 / in_parameters['reg_alpha']) * -1) + 1 
        left = min(parameter_ranges['reg_alpha']) 
        
        if left == right:
            new_value = np.power(10, left)
        else:
            new_value = np.power(10, float(np.random.randint(left,right)))
        
        new_value = round(new_value, 4)
        out_parameters.update({'reg_alpha':new_value})
        
        #'lambda' - decrease to reduce model complexity
        right =  (np.log10(1.0 / in_parameters['reg_lambda']) * -1) + 1 
        left = min(parameter_ranges['reg_lambda']) 
        
        if left == right:
            new_value = np.power(10, left)
        else:
            new_value = np.power(10, float(np.random.randint(left,right)))
        
        out_parameters.update({'reg_lambda': round(new_value,4)})
        

    #pprint.pprint(in_parameters)
    #pprint.pprint(out_parameters)
    
    return out_parameters



    
    
    
    
def meter(result, context, threshold, parameter_ranges, human_accuracy = 1.0):
    train_accuracy = 1.0 - result[0]
    valid_accuracy = 1.0 - result[1]
    test_accuracy = 1.0 - result[2]
    
    


    bias = abs(human_accuracy - train_accuracy)
    variance = abs(valid_accuracy - train_accuracy)
    
#     print(train_accuracy,valid_accuracy,test_accuracy)     
    #print(bias,variance, threshold)
    
    if bias < threshold:
        if variance < threshold:
            return('tuned') 
        else:
            return('High Variance')
    else:
        return('High Bias')    
    

    
    
    
def ngtuner( datasets, context, threshold, parameter_ranges, training_results, interval = 0 ):
    """Andrew Ng's recipe
    """
    intervals =[0]
    pickled = pickler(context['pickle'], intervals, 'tuner intervals')
    def callback(env):

        if interval > 0 and env.iteration > 0 and env.iteration % interval == 0:

            
            intervals.append(int(env.iteration))
            pickled = pickler(context['pickle'], intervals, 'tuner intervals')
            bst, i, n = env.model, env.iteration, env.end_iteration
            train_err= training_results['Train']['merror'][env.iteration-1]
            valid_err= training_results['Valid']['merror'][env.iteration-1]
            test_err= training_results['Test']['merror'][env.iteration-1]
            
            #pprint.pprint(training_results)
            #print(train_err,test_err, valid_err)
            #print(len(training_results),i,n)
            

    
            avg = [train_err, valid_err, test_err]

            write_dict({'train, valid, test': avg}, context['summary'],' merrors')

        
            diag = meter(avg, context, threshold, parameter_ranges, human_accuracy = 1.0)
            write_dict({'diagnosis': diag}, context['summary'],' Diagnosis')
        
            pickled = pickler(context['pickle'])
            parameters = pickled['optimal parameters']
            #pprint.pprint(parameters)
            new_params = tuner(diag, parameters, parameter_ranges, context)
            pickled = pickler(context['pickle'], new_params, 'optimal parameters')
        
            #pprint.pprint(new_params)
            
            write_dict({'iteration': env.iteration}, context['summary'],' Tuning Iteration')
            write_dict(new_params, context['summary'],' Updated Parameters')

            bst.set_param(new_params)
            

    
    callback.before_iteration = True
    


    
    #pprint.pprint(training_results)
    return callback


    
    
    
def modelfit(params, datasets, labels, context, title, parameter_ranges, interval = 0, threshold = 0.10, useTrainCV=True, early_stopping_rounds=20):
      
    try:
        train_dataset= datasets[0]
        train_labels = labels[0]
    
    except Exception as e:
        print('Unable to save data to load training samples', e)
        raise
    
    valid_dataset = datasets[1]
    test_dataset = datasets[2]
    
   
    valid_labels = labels[1]
    test_labels = labels[2]

    
    run_stats={}
    optimal_boosters = 0
    #num_class = num_labels
    
    
    if useTrainCV:

        
        xgb_param = params
        intervals =[interval]
        pickled = pickler(context['pickle'], intervals, 'tuner intervals')
    
        write_dict({'        ' : title}, context['summary'],' ')
        write_dict(xgb_param, context['summary'],' Initial Parameters')
        #xgb_param.update({'num_class': num_class})
        updated_pickle = pickler(context['pickle'], xgb_param, 'optimal parameters')


        xgtrain = xgb.DMatrix(train_dataset,label=train_labels)
        xgvalid = xgb.DMatrix(valid_dataset,label=valid_labels)
        xgtest = xgb.DMatrix(test_dataset,label=test_labels)
        
        xgdataset = [(xgtrain, 'Train'), (xgtest, 'Test'), (xgvalid, 'Valid') ]
        training_results ={}
        

        
        cv_start_time = time.time()
        if interval ==0:
            cvresult = xgb.train(xgb_param, xgtrain, num_boost_round=params['n_estimators'], evals = xgdataset,
                          evals_result = training_results, early_stopping_rounds=early_stopping_rounds,verbose_eval=10)
        
               
                
        else:
            cvresult = xgb.train(xgb_param, xgtrain, num_boost_round=params['n_estimators'], evals = xgdataset,
                          evals_result = training_results, early_stopping_rounds=early_stopping_rounds,verbose_eval=interval,
                          callbacks=[ngtuner( xgdataset, context, threshold, parameter_ranges, training_results, interval)])
        
        cv_end_time = time.time()
        
        #pprint.pprint(training_results)
        

        
        cv_time_raw = cv_end_time - cv_start_time
        cv_time = time.strftime("%H:%M:%S s",time.gmtime(cv_time_raw))

        

        
 
        #alg.set_params(n_estimators = cvresult.shape[0])
        optimal_boosters = cvresult.attr('best_iteration')
        #optimal_boosters = 10

        
    #Fit the algorithm on the data
    fit_start_time =time.time()
    #alg.fit(train_dataset, train_labels,eval_metric=metrics)
    fit_end_time =time.time()
    
    fit_time_raw = fit_end_time - fit_start_time
    fit_time = time.strftime("%H:%M:%S s",time.gmtime(fit_time_raw))

   
    #Predict training and validation set:
    predict_start_time = time.time()
    #dtrain_predictions = alg.predict(train_dataset)
    #dvalid_predictions = alg.predict(valid_dataset)
    #dtest_predictions = alg.predict(test_dataset)
    predict_end_time = time.time()
    
    predict_time_raw = predict_end_time - predict_start_time
    predict_time = time.strftime("%H:%M:%S s",time.gmtime(predict_time_raw))

    
    write_dict({'train time': cv_time, 'fit time': fit_time, 'predict time': predict_time}, context['summary'],'Run Times')

        
     #Print model report:
    #acc_score_train = accuracy_score(train_labels, dtrain_predictions)
    #acc_score_valid = accuracy_score(valid_labels, dvalid_predictions)
    #acc_score_test = accuracy_score(test_labels, dtest_predictions)
    #print ("\nModel Report")
    #print ("Accuracy : {0:.5f}".format(acc_score_test)) 
    #print ("Optimal Boosters : {}".format(optimal_boosters)) 
    
    #run_stats.update({'Train Accuracy': acc_score_train})
    #if acc_score_valid: run_stats.update({'Validation Accuracy': acc_score_valid})
    #if acc_score_test: run_stats.update({'Test Accuracy': acc_score_test})

    pickled = pickler(context['modelpickles'], cvresult, 'model')

    
    feat_imp_ser = pd.Series(cvresult.get_fscore()).head(10).sort_values(ascending=False)
    feat_dict = feat_imp_ser.to_dict()
    # run_stats.update({'Feature Importance Score': feat_dict})
     
    write_dict(run_stats, context['summary'], 'Results')
    pickled = pickler(context['pickle'], run_stats, 'model results')

        
    intervals = pickled['tuner intervals']
    #intervals.append(int(optimal_boosters))
    #print(intervals)

    plotCV(cvresult, int(optimal_boosters), context, intervals, training_results, title)
    
    
    ##########Book keeping - update optimal parameters in dictionary with new boosters
    
    pickled = pickler(context['pickle'])
    parameters = pickled['optimal parameters']
    parameters['n_estimators'] = int(optimal_boosters)

    #native xgboost requires num_class, scikit_learn doesnt like it
    #del parameters['num_class']

    #update with results
    #parameters.update({'n_estimators': optimal_boosters})

    updated_pickle = pickler(context['pickle'], parameters, 'optimal parameters')
    updated_pickle = pickler(context['pickle'], run_stats, 'run results')
    write_dict(parameters, context['summary'],' Final Parameters')
    
    return updated_pickle
    ########## End Book Keeping
 


def plotCV(cvresult, optimal_boosters, context, intervals, training_results, title ='accuracy score by #estimators', ylim=(0.5,1)):
    # ylim=(0.8,1.01)
    
    plt.rcParams['figure.figsize'] = (20,10)
    plt.style.use('seaborn-colorblind')
    sns.set_style("whitegrid")
    watermark = mpimg.imread('../images/current_logo_gray.png')
    
    

    
    #cvresult_df = pd.DataFrame(cvresult)
    x_values = len(training_results['Train']['merror'])
    test_error = training_results['Test']['merror']
    #test_std = cvresult_df.iloc[:,1].tolist()
    
    train_error = training_results['Train']['merror']
    valid_error = training_results['Valid']['merror']
    #train_std = cvresult_df.iloc[:,3].tolist()
    
    x_values_int= [None]* x_values
    test_acc_float= [None]*len(test_error)
    #test_std_float= [None]*len(x_values)
    train_acc_float= [None]*len(train_error)
    valid_acc_float= [None]*len(valid_error)
    #train_std_float= [None]*len(x_values)
    
    for i in range(x_values):
        x_values_int[i] = i
        test_acc_float[i] = 1 - float(test_error[i])
        #test_std_float[i] = float(test_std[i])
        train_acc_float[i] = 1 - float(train_error[i])
        valid_acc_float[i]= 1 - float(valid_error[i])
        #train_std_float[i] = float(train_std[i])
        
      
    fig = plt.figure()
    plt.xlabel('number of estimators')
    plt.ylabel('accuracy')
    #plt.ylim(0.7,1,1)

    plt.plot(x_values_int,
         train_acc_float,
         label='Training Score',
         color = 'r')

    plt.plot(x_values_int,
         test_acc_float,
         label='Test Score',
         color = 'g')
    
    plt.plot(x_values_int,
         valid_acc_float,
         label='Validation Score',
         color = 'b')




    plt.axhline(y = 1, color='k',linewidth=1, ls ='dashed')
    plt.axhline(y = valid_acc_float[int(optimal_boosters)], color='k',linewidth=2, ls ='dashed')
    plt.axvline(x = int(optimal_boosters), color='k',linewidth=2, ls ='dashed')
    #print(intervals[len(intervals)-1])
    
    for interval in intervals:
        
        plt.axvline(x = interval, linewidth=1, ls ='dashed')

    
#     plt.plot(optimal_boosters, float(accuracy_train), 'b^', label = 'Train Accuracy: ' + str(accuracy_train))
#     plt.plot(optimal_boosters, float(accuracy_valid), 'm^', label = 'Valid Accuracy: ' + str(accuracy_valid))
#     plt.plot(optimal_boosters, float(accuracy_test), 'g^', label = 'Test Accuracy: '+ str(accuracy_test))

    
    plt.legend(loc = 'best')
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    
    x_axis_range = plt.xlim()
    y_axis_range = plt.ylim()

    
    imgplot = plt.imshow(watermark, aspect = 'auto', extent=(x_axis_range[0], x_axis_range[1],  y_axis_range[0],  y_axis_range[1]), zorder= - 1, alpha =0.1)

    now = time.strftime("%H%M%S",time.gmtime())
    save_file = context['plot path'] + now + '.png'
    plt.text(x_axis_range[0], y_axis_range[0], save_file, color='gray', fontsize=8)
    
    plt.show()
    
    fig.savefig(save_file, bbox_inches='tight')
    pickled = pickler(context['pickle'], save_file, 'accuracy plot')





def remove_duplicates(inlist):
    outlist =[]
    for i in inlist:
        if i not in outlist:
            outlist.append(float(i))
    
    return outlist
    

  
    
    

def show_random_samples(image_size, dataset, labels, description, context, rows = 1, cols=10):
    
    unique, counts = np.unique(labels, return_counts=True)
    #print(unique,counts)
    font = {'family': 'monospace',
        'color':  '#351c4d',
        'weight': 'normal',
        'size': 20,
        }
    
    label_list = list(string.ascii_uppercase)
    plt.rcParams['figure.figsize'] = (20,14)
    plt.style.use('seaborn-colorblind')
    #sns.set_style("whitegrid")
    watermark = mpimg.imread('../images/current_logo_gray.png')
    footer_height =2
    footer_width = 4
    
    
    fig = plt.figure()
    counter = 1
    
    for row in range(rows):
  
            for col in range(cols):
                pick =   np.where(labels == col)[0] 
                random_pick = np.random.randint(len(pick))
                sample_idx = pick[random_pick]
                #print(sample_idx)
                
                #sample_idx = np.random.randint(len(dataset)) 
                sample_label = labels[sample_idx]  
        
                    
                sample_image = dataset[sample_idx, :] 
                a=fig.add_subplot(rows + footer_height, cols, counter)
                sample_image = sample_image.reshape(image_size, image_size)
                #plt.axis('off')
                plt.imshow(sample_image)
                a.set_title(label_list[sample_label], fontsize=12, weight = 'bold',color = 'r')
                counter+=1
                    
                
    
    
   
    for col in range(cols):
                pick =   np.where(labels == col)[0] 
                random_pick = np.random.randint(len(pick))
                sample_idx = pick[random_pick]
                sample_label = labels[sample_idx]      
                sample_image = dataset[sample_idx, :] 
                b=fig.add_subplot(rows + footer_height, cols, counter)
                sample_image = sample_image.reshape(image_size, image_size)
                #plt.axis('off')
                plt.imshow(sample_image,cmap='Greys_r')
                plt.tight_layout()
                b.set_title(counts[sample_label], fontsize=15, weight = 'bold',color = '#351c4d')
                counter+=1
    
    logo_footer= fig.add_subplot(rows + footer_height,footer_width,(rows+ footer_height)* footer_width)
    x_axis_range = plt.xlim()
    y_axis_range = plt.ylim()

    #plt.axis('off')
    #sns.set_style("whitegrid")
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='off',      # ticks along the bottom edge are off
    right='off',         # ticks along the top edge are off
    labelleft='off') # labels along the bottom edge are off
    
    imgplot = plt.imshow(watermark, aspect = 'auto', extent=(x_axis_range[0], x_axis_range[1],  y_axis_range[0],  y_axis_range[1]), zorder= - 1, alpha =0.3)
  
    
    
    
    title_footer= fig.add_subplot(rows + footer_height,2,(rows+ footer_height)* 2-1)
    plt.axis('off')
    x_axis_range = plt.xlim()
    y_axis_range = plt.ylim()
    plt.text(0, 0, description + str(dataset.shape), va='center', fontdict=font, fontsize=40)
    
    now = time.strftime("%H%M%S",time.gmtime())
    save_file = context['plot path'] + now + '.png'
    
    plt.text(x_axis_range[0], y_axis_range[0], save_file, color='gray', fontsize=8)
    
    plt.show()
    fig.savefig(save_file, bbox_inches='tight')
    
    




    
    

