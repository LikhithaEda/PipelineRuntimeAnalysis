import datetime
import json
from numpy import NaN
import requests
import sys
import pandas as pd
from pandas import json_normalize
import json 

def getData(topic,delta):
    startURL = "https://datagrepper.engineering.redhat.com/raw?topic=/topic/"
    deltaURL = "&delta="
    url = startURL + topic + deltaURL + delta
    request =requests.get(url, verify="/etc/pki/ca-trust/source/anchors/2015-RH-IT-Root-CA.pem")
    if request.status_code == 200: #200 means request fulfilled
        return request.json()
        #return request.text
    else:
        raise Exception("Query failed to run")

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
    return StageData_df

def getFirstStageData(delta):
    """VirtualTopic.eng.ci.redhat-container-image.pipeline.running: this data will tell when a new pipeline is running and the timestamp of when it started"""
    result = getData('VirtualTopic.eng.ci.redhat-container-image.pipeline.running', delta)

    with open('FirstStage_epoch_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

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
    #print("start time")
    #print(FirstStageData_df)
    #convert to human readble time
    FirstStageData_df['startTime'] = convertEpoch(FirstStageData_df['startTime'])
    FirstStageData_df.to_csv("FirstStageData.csv", encoding='utf-8', index=False)
    #print(FirstStageData_df)
    return FirstStageData_df

def getFinalStageData(delta):
    """VirtualTopic.eng.ci.redhat-container-image.pipeline.complete: data for when final stage execution is complete"""
    result = getData('VirtualTopic.eng.ci.redhat-container-image.pipeline.complete', delta)
    with open('FinalStage_epoch_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
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

    #print("end time")
    #print(FinalStageData_df)
    
    #convert to human readble time
    FinalStageData_df['endTime'] = convertEpoch(FinalStageData_df['endTime'])
    FinalStageData_df.to_csv("FinalStageData.csv", encoding='utf-8', index=False)
    return FinalStageData_df

def calculateRuntime(start, end):
  time_diff = (start - end).total_seconds() / 60 #return diff in mins
  return int(time_diff)

def createGraph(df):
    return None

def main(argv):
    """read command line for delta configuration"""
    delta = "19500" #getting delta from run command sys.argv[1]
    #get the data
    FirstStage = getFirstStageData(delta)
    FinalStage = getFinalStageData(delta)
    #stageData = getStageData(delta)

    #create a main dataframe for pipeline executions
    #runtimes_df = pd.concat([FirstStage['timestamp'], FinalStage['timestamp']], axis=1, keys=['start time', 'end time'])
    
    #merge Firststage data and finalstage data based on nvr
    runtimes_df= pd.merge(FirstStage, FinalStage, on='nvr+pipelineID')
    #print(runtimes_df)
    
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
        
    runtimes_df.to_csv("runtime.csv", encoding='utf-8', index=False)
    #print(runtimes_df)




if __name__ == "__main__":
   main(sys.argv[1:])

