import datetime
import json
from numpy import NaN
import requests
import sys
import pandas as pd
from pandas import json_normalize
import json
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

def getData(topic,delta):
    # startURL = "https://datagrepper.engineering.redhat.com/raw?topic=/topic/"
    print(f"Getting data for topic {topic}")
    messages = []
    page = 1
    rows_per_page = 50
    total = None
    while total is None or page <= ceil(total / rows_per_page):
        url = "https://datagrepper.engineering.redhat.com/raw"
        params = {
            'delta': delta,
            'topic': '/topic/' + topic,
            'page': str(page),
            'rows_per_page': str(rows_per_page)
        }

        response = requests.get(url, params=params, verify="/etc/pki/ca-trust/source/anchors/2015-RH-IT-Root-CA.pem")
        if response.status_code == 200: #200 means request fulfilled
            response_json = response.json()
            total = response_json['total']
            print(f"Got page {page} / {ceil(total / rows_per_page)}")
            messages += response_json['raw_messages']
            page += 1
        else:
            raise Exception("Query failed to run")
    return {'raw_messages': messages}

def convertEpoch(col):
    """Convert timestamp from epoch to datetime"""
    timeCol = []
    for i in col:
        if ( i != NaN ):
            time = datetime.datetime.fromtimestamp(i)
            timeCol.append(time)
        else:
            timeCol.append(NaN)
    col = timeCol
    return col

def getStageData(delta):
    """VirtualTopic.eng.ci.redhat-container-image.pipeline.stage: beginning and end of a single stage"""
    result = getData('VirtualTopic.eng.ci.redhat-container-image.pipeline.stage', delta)
    #extract stage info
    y = []
    for index in range(0,len(result['raw_messages'])):
        y.append(result['raw_messages'][index]['msg']['stage'])
    StageData_df = pd.DataFrame(y) 
    print(StageData_df)
    return StageData_df

def getFirstStageData(delta):
    """VirtualTopic.eng.ci.redhat-container-image.pipeline.running: this data will tell when a new pipeline is running and the timestamp of when it started"""
    result = getData('VirtualTopic.eng.ci.redhat-container-image.pipeline.running', delta)

    """
    with open('FirstStage_epoch_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    """

    #get timestamp
    y = []
    for index in range(0,len(result['raw_messages'])):
        y.append(result['raw_messages'][index]['timestamp'])
    #create dataframe
    FirstStageData_df = pd.DataFrame(y) 
    FirstStageData_df.columns = ['startTime']

    #get nvr info
    x = []
    for index in range(0,len(result['raw_messages'])):
        x.append(result['raw_messages'][index]['msg']['artifact']['nvr'])
    FirstStageData_df['nvr'] = pd.DataFrame(x)

    #get pipeline id
    z = []
    for index in range(0,len(result['raw_messages'])):
        z.append(result['raw_messages'][index]['msg']['pipeline']['id'])
    FirstStageData_df['pipelineID'] = pd.DataFrame(z)

    #combine nvr+pipeline id columns 
    FirstStageData_df['nvr+pipelineID'] = FirstStageData_df['nvr'] + FirstStageData_df['pipelineID']
   
    #convert to human readble time
    FirstStageData_df['startTime'] = convertEpoch(FirstStageData_df['startTime'])
    #FirstStageData_df.to_csv("FirstStageData.csv", encoding='utf-8', index=False)

    return FirstStageData_df

def getFinalStageData(delta):
    """VirtualTopic.eng.ci.redhat-container-image.pipeline.complete: data for when final stage execution is complete"""
    result = getData('VirtualTopic.eng.ci.redhat-container-image.pipeline.complete', delta)
    """
    with open('FinalStage_epoch_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    """
    #get timestamp
    y = []
    for index in range(0,len(result['raw_messages'])):
        y.append(result['raw_messages'][index]['timestamp'])
    
    #create dataframe
    FinalStageData_df = pd.DataFrame(y) 
    FinalStageData_df.columns = ['endTime']

    #get nvr info
    x = []
    for index in range(0,len(result['raw_messages'])):
        x.append(result['raw_messages'][index]['msg']['artifact']['nvr'])
    FinalStageData_df['nvr'] = pd.DataFrame(x)

    #get pipeline id
    z = []
    for index in range(0,len(result['raw_messages'])):
        z.append(result['raw_messages'][index]['msg']['pipeline']['id'])
    FinalStageData_df['pipelineID'] = pd.DataFrame(z)

    #combine nvr+pipeline id columns 
    FinalStageData_df['nvr+pipelineID'] = FinalStageData_df['nvr'] + FinalStageData_df['pipelineID']
    
    #convert to human readble time
    FinalStageData_df['endTime'] = convertEpoch(FinalStageData_df['endTime'])
    #FinalStageData_df.to_csv("FinalStageData.csv", encoding='utf-8', index=False)
    return FinalStageData_df

def calculateRuntime(start, end):
  time_diff = (end - start).total_seconds() / 60 #return diff in mins
  return int(time_diff)

def addlabels(x, y):
    """create labels for bars in bar chart"""

    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')

def createGraph(x_col, y_col):
    
    fig, ax = plt.subplots()
    x_pos = np.arange(len(x_col))  # <--
    plt.bar(x_pos, y_col)
    plt.xticks(x_pos, x_col)  # <--
    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()
    ax.xaxis_date()
    addlabels(x_col, y_col)
    plt.xlabel("Dates")
    plt.ylabel("Run Time in Minutes")
    plt.title("Average Run Times")
    plt.savefig('AvgRunTimes.png', dpi=400)
    plt.show()

def main(argv):
    """read command line for delta configuration"""
    delta =  sys.argv[1] #getting delta from run command 
    
    #get pipeline runtimes
    #if (sys.argv[2] == 1):
    #get the data
    FirstStage = getFirstStageData(delta)
    FinalStage = getFinalStageData(delta)
    #merge Firststage data and finalstage data based on nvr
    runtimes_df= pd.merge(FirstStage, FinalStage, on='nvr+pipelineID')
        
    #calculate run time
    runtime = []
    for index, row in runtimes_df.iterrows():
        if(row.loc['startTime'] != "NaN" and row.loc['endTime'] != "NaN"):
            runtime_ = calculateRuntime(row.loc['startTime'] , row.loc['endTime'])
            runtime.append(runtime_)
        else:
            runtime.append("NaN")
    #add column to df
    runtimes_df['run time(mins)'] = runtime
            
        #runtimes_df.to_csv("runtime.csv", encoding='utf-8', index=False)
        #print(runtimes_df)

        #create a year-month column based on runtime end date
        #group by the year-month column and find runtime avgs
    runtimes_df['endTime_YM'] = pd.to_datetime(runtimes_df['endTime']).dt.to_period('D')
    new_df = runtimes_df.filter(['endTime_YM', 'run time(mins)'], axis=1)
    group_mean = new_df.groupby('endTime_YM')['run time(mins)'].mean()
    mean_df = group_mean.reset_index()

    mean_df['run time(mins)'] = mean_df['run time(mins)'].astype(int)

        #create a graph for avg pipeline runtime
    createGraph(mean_df['endTime_YM'], mean_df['run time(mins)'])
 
    #get stage runtimes
    #elif (sys.argv[2] == 2):
    stageData = getStageData(delta)
    print(stageData)
    
    #create a main dataframe for pipeline executions
    #runtimes_df = pd.concat([FirstStage['timestamp'], FinalStage['timestamp']], axis=1, keys=['start time', 'end time'])
    
    #merge Firststage data and finalstage data based on nvr
    


if __name__ == "__main__":
   main(sys.argv[1:])