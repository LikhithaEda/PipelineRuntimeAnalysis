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


def getData(topic, delta):

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

        response = requests.get(url, params=params)
        if response.status_code == 200:  # 200 means request fulfilled
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
        if i != NaN:
            time = datetime.datetime.fromtimestamp(i)
            timeCol.append(time)
        else:
            timeCol.append(NaN)
    col = timeCol
    return col


def getStageData(delta):
    """VirtualTopic.eng.ci.redhat-container-image.pipeline.stage:
    beginning and end of a single stage"""
    result = getData(
                     'VirtualTopic.eng.ci.redhat-container-image.pipeline.stage',
                     delta)
    # extract stage info, nvr, pipeline id
    y = []
    x = []
    z = []
    t = []
    p = []
    for index in range(0, len(result['raw_messages'])):
        y.append(result['raw_messages'][index]['msg']['stage'])
        x.append(result['raw_messages'][index]['msg']['artifact']['nvr'])
        z.append(result['raw_messages'][index]['msg']['pipeline']['id'])
        t.append(result['raw_messages'][index]['timestamp'])
        p.append(result['raw_messages'][index]['msg']['pipeline']['name'])
    StageData_df = pd.DataFrame(y)
    StageData_df['nvr'] = pd.DataFrame(x)
    StageData_df['pipelineID'] = pd.DataFrame(z)
    StageData_df['timestamp'] = pd.DataFrame(t)
    StageData_df['pipelineName'] = pd.DataFrame(p)

    # combine nvr+pipeline id columns
    StageData_df['nvr+pipelineID'] = StageData_df['nvr'] + StageData_df['pipelineID']

    # convert to human readble time
    StageData_df['timestamp'] = convertEpoch(StageData_df['timestamp'])

    return StageData_df


def getFirstStageData(delta):
    """VirtualTopic.eng.ci.redhat-container-image.pipeline.running:
    this data will tell when a new pipeline is running and the timestamp of when it started"""
    result = getData('VirtualTopic.eng.ci.redhat-container-image.pipeline.running', delta)

    """
    with open('FirstStage_epoch_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    """

    # get timestamp, nvr, pipeline id
    y = []
    x = []
    z = []
    p = []
    for index in range(0, len(result['raw_messages'])):
        y.append(result['raw_messages'][index]['timestamp'])
        x.append(result['raw_messages'][index]['msg']['artifact']['nvr'])
        z.append(result['raw_messages'][index]['msg']['pipeline']['id'])
        p.append(result['raw_messages'][index]['msg']['pipeline']['name'])
    # create dataframe
    FirstStageData_df = pd.DataFrame(y)
    FirstStageData_df.columns = ['startTime']
    FirstStageData_df['nvr'] = pd.DataFrame(x)
    FirstStageData_df['pipelineID'] = pd.DataFrame(z)
    FirstStageData_df['pipelineName'] = pd.DataFrame(p)

    # combine nvr+pipeline id columns
    FirstStageData_df['nvr+pipelineID'] = FirstStageData_df['nvr'] + FirstStageData_df['pipelineID']

    # drop pipeline id and nvr columns
    # FirstStageData_df.drop['nvr', 'pipelineID']

    # convert to human readble time
    FirstStageData_df['startTime'] = convertEpoch(FirstStageData_df['startTime'])
    # FirstStageData_df.to_csv("FirstStageData.csv", encoding='utf-8', index=False)

    return FirstStageData_df


def getFinalStageData(delta):
    """VirtualTopic.eng.ci.redhat-container-image.pipeline.complete:
    data for when final stage execution is complete"""
    result = getData('VirtualTopic.eng.ci.redhat-container-image.pipeline.complete', delta)
    """
    with open('FinalStage_epoch_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    """
    # get timestamp, nvr, pipeline id
    y = []
    x = []
    z = []
    p = []
    for index in range(0, len(result['raw_messages'])):
        y.append(result['raw_messages'][index]['timestamp'])
        x.append(result['raw_messages'][index]['msg']['artifact']['nvr'])
        z.append(result['raw_messages'][index]['msg']['pipeline']['id'])
        p.append(result['raw_messages'][index]['msg']['pipeline']['name'])
    # create dataframe
    FinalStageData_df = pd.DataFrame(y)
    FinalStageData_df.columns = ['endTime']
    FinalStageData_df['nvr'] = pd.DataFrame(x)
    FinalStageData_df['pipelineID'] = pd.DataFrame(z)
    FinalStageData_df['pipelineName'] = pd.DataFrame(p)

    # combine nvr+pipeline id columns
    FinalStageData_df['nvr+pipelineID'] = FinalStageData_df['nvr'] + FinalStageData_df['pipelineID']

    # convert to human readble time
    FinalStageData_df['endTime'] = convertEpoch(FinalStageData_df['endTime'])
    # FinalStageData_df.to_csv("FinalStageData.csv", encoding='utf-8', index=False)
    return FinalStageData_df


def calculateRuntime(start, end):
    time_diff = (end - start).total_seconds() / 60  # return diff in mins
    return int(time_diff)


def addlabels(x, y):
    """create labels for bars in bar chart"""

    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')


def createBarGraph(x_col, y_col, x_label, y_label, title, filename):

    fig, ax = plt.subplots()
    x_pos = np.arange(len(x_col))  # <--
    plt.bar(x_pos, y_col)
    plt.xticks(x_pos, x_col)  # <--
    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()
    ax.xaxis_date()
    addlabels(x_col, y_col)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def main(argv):
    """read command line for delta configuration"""
    delta = sys.argv[1]
    # get the data
    FirstStage = getFirstStageData(delta)
    FinalStage = getFinalStageData(delta)

    # merge Firststage data and finalstage data based on nvr+pipelineID
    runtimes_df = pd.merge(FirstStage, FinalStage, on='nvr+pipelineID')
    # create only one column of pipeline name
    runtimes_df.drop('pipelineName_y', axis=1, inplace=True)
    runtimes_df.rename({'pipelineName_x': 'pipelineName'}, axis=1, inplace=True)
    # calculate run time
    runtime = []
    for index, row in runtimes_df.iterrows():
        if(row.loc['startTime'] != "NaN" and row.loc['endTime'] != "NaN"):
            runtime_ = calculateRuntime(row.loc['startTime'], row.loc['endTime'])
            runtime.append(runtime_)
        else:
            runtime.append("NaN")
    # add column to df
    runtimes_df['run time(mins)'] = runtime

    # create a year-month column based on runtime end date
    # group by the year-month column and find runtime avgs
    runtimes_df['endTime_YM'] = pd.to_datetime(runtimes_df['endTime']).dt.to_period('D')
    new_df = runtimes_df.filter(['pipelineName', 'endTime_YM', 'run time(mins)'], axis=1)
    group_mean = new_df.groupby('endTime_YM')['run time(mins)'].mean()
    mean_df = group_mean.reset_index()

    mean_df['run time(mins)'] = mean_df['run time(mins)'].astype(int)

    # create a graph for avg pipeline runtime
    createBarGraph(mean_df['endTime_YM'], mean_df['run time(mins)'], "Dates",
                   "Run Time in Minutes", f"Average Run Times in Timeframe of {delta} Seconds",
                   "avgRunTimes.png")

    """ Create an analysis on avg runtimes based on pipeline names given the time period"""

    Pgroup_mean = new_df.groupby('pipelineName')['run time(mins)'].mean()
    Pmean_df = Pgroup_mean.reset_index()

    Pmean_df['run time(mins)'] = Pmean_df['run time(mins)'].astype(int)
    createBarGraph(Pmean_df['pipelineName'], Pmean_df['run time(mins)'],
                   "Pipeline", "Runtime in Minutes",
                   f"Runtime Averages for each Pipeline in Timeframe of {delta} Seconds",
                   "pipelineRuntimes.png")

    """ Create an analysis on avg runtime for each stage in pipeline"""

    stageData = getStageData(delta)
    # split stageData based on status
    stageComplete = stageData[stageData['status'] != 'running']
    stageComplete = stageComplete[stageComplete['runtime'] != 'NaN']

    # calculate averages based on stage type, nvr, pipelineID
    stageComplete['timestampYM'] = pd.to_datetime(stageComplete['timestamp']).dt.to_period('D')
    stage_df = stageComplete.filter(['pipelineName', 'name', 'runtime', 'timestampYM'], axis=1)
    stage_df[["runtime"]] = stage_df[["runtime"]].apply(pd.to_numeric)

    Sgroup_mean = stage_df.groupby('name')['runtime'].mean()
    Smean_df = Sgroup_mean.reset_index()
    Smean_df['runtime'] = Smean_df['runtime'].astype(int)
    createBarGraph(Smean_df['name'], Smean_df['runtime'], "Stage Name", "Runtime in Seconds", f"Average Runtime for each Stage in Timeframe of {delta} seconds", "stageAvg.png")


if __name__ == "__main__":
    main(sys.argv[1:])
