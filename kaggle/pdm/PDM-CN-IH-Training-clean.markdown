---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.6
  nbformat: 4
  nbformat_minor: 5
  papermill:
    default_parameters: {}
    duration: 1258.31434
    end_time: "2023-11-04T11:48:20.025547"
    environment_variables: {}
    input_path: \_\_notebook\_\_.ipynb
    output_path: \_\_notebook\_\_.ipynb
    parameters: {}
    start_time: "2023-11-04T11:27:21.711207"
    version: 2.3.4
---

::: {#c0c0da01 .cell .markdown}
# 导入Python库
:::

::: {#6f35044a-21a5-49ff-9137-48b15fd4ce00 .cell .code}
``` python
%pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost graphviz joblib requests awscli pyarrow
```
:::

::: {#fc94b0a0 .cell .markdown}
# 初始化环境
:::

::: {#06e2d4dd-b27b-41f0-9ab0-444f3b8b1600 .cell .code}
``` python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
from pathlib import Path
import os, json, requests, datetime, tempfile, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup():
    font_path = Path("simhei.ttf").resolve()
    print(f"尝试加载字体路径：{font_path}")  # 输出完整路径，检查是否正确
    # 检查该路径是否存在
    if not font_path.exists():
      print(f"错误：字体文件不存在于 {font_path}")
    else:
      print("字体文件存在。")
    # 然后尝试加载...
    fm.fontManager.addfont(font_path)
    fm.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore')
    # 设置图表样式

    sns.set_palette("viridis")

setup()
```
:::

::: {#90d378be-210b-43e6-9f1e-50a6a967c497 .cell .markdown}
# Insights Hub 读写Asset和IDL的函数实现
:::

::: {#70e6dc49-343c-45cb-8cb3-2463abe7c0f5 .cell .code}
``` python
# Define a function that performs IoT data reads
def read_iot(entity_id = "<<iot_entity_id_GUID>>",
           aspect_name = "<<aspect_name>>",
           tenant = "tenantname",
           max_results = 2000, #max is 2000
           from_dt = "2020-06-01T13:09:37.029Z", 
           to_dt = "2020-07-01T08:02:27.962Z",
           variable = "pressure",
           sort = "asc"):

    print(from_dt)
    print(to_dt)
    if variable is not None:
        url = "?from=" + from_dt + "&to=" + to_dt + "&sort=" + sort + "&limit=" + str(max_results) + "&select=" + variable
    else:
        url = "?from=" + from_dt + "&to=" + to_dt + "&sort=" + sort + "&limit=" + str(max_results)

    # this is the IoT Timeseries API base URL
    TSpath = 'iottimeseries/v3/timeseries'

    # this is the Predictive Gateway URL that handles authentication for your API calls
    gw = os.environ['GATEWAY_ENDPOINT'] + '/gateway/'
    headers = {
        'Content-Type': 'application/json'
    }
    iot_url = gw + TSpath + "/" + entity_id + "/" + aspect_name + url
    response = requests.get(iot_url, headers=headers)
    return response
```
:::

::: {#8ed53481-1895-4d32-bf59-6bd959a2507a .cell .code}
``` python
# Define a function that performs IoT data writes
def write_to_iot(entity_id = "<<iot_entity_id_GUID>>",
    aspect_name = "<<aspect_name>>",
    tenant = "tenantname",
    data_records = "<<json_data_records>>"):

    # this is the IoT Timeseries API base URL
    TSpath = 'iottimeseries/v3/timeseries'

    # this is the Predictive Gateway URL that handles authentication for your API calls
    gw = os.environ['GATEWAY_ENDPOINT'] + '/gateway/'
    headers = {
        'Content-Type': 'application/json'
    }
    iot_url = gw + TSpath + "/" + entity_id + "/" + aspect_name

    # data_records = [
    #    { "_time": current_time, "temperature": 25.3, "pressure": 1.2 },
    #    { "_time": current_time, "temperature": 25.5, "pressure": 1.3 }]
    
    response = requests.put(iot_url, headers=headers, data=json.dumps(data_records))
    return response
```
:::

::: {#e5f6bcdd-1a80-4ed3-a548-8e590fc0e0c6 .cell .code}
``` python
# 设定读取数据的范围
#start = datetime.datetime.utcnow() - datetime.timedelta(days=70)
#end = start + datetime.timedelta(days=30)

# 读取最近1小时的数据
hours = 3
start = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
end = datetime.datetime.utcnow()
```
:::

::: {#e61776c5-2dcb-4e31-ad54-95e63e312426 .cell .code}
``` python
# read IoT data 的示例
response = read_iot(entity_id = "772d234e2f3e4046bc731704ab732c8e",
    aspect_name = " zl_reflow_oven",
    tenant = "presiot",
    max_results = 2000, #max is 2000
    from_dt = start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
    to_dt = end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
    sort = "asc",
    variable = None)

if response.status_code == 200:
    f = tempfile.TemporaryFile()
    f.write(response.content)
    f.seek(0)

    # we read the IoT data into a Pandas DataFrame
    data = pd.read_json(io.BytesIO(f.read()))
    f = tempfile.TemporaryFile()
    f.write(response.content)
    f.seek(0)
    print(data.shape)
else:
    print(response.status_code)
    print(response.content)
```
:::

::: {#cb401139-6147-4def-9170-891e666025b1 .cell .code}
``` python
# 上传到IDL
# The IDL endpoint
dlpath = '/datalake/v3/generateAccessToken'

# The base gateway that provides transparent authentication
gw = os.environ['GATEWAY_ENDPOINT'] + '/gateway/'

headers = {
'Content-Type': 'application/json'
}   
payload="{ \"subtenantId\":\"\" } "
dl_url = gw + dlpath

# Perform the request
response = requests.post(dl_url, data=payload, headers=headers)

# The response should be a JSON object
idlSession = json.loads(response.text)

# Update the environments with the temporary credentials
os.environ["AWS_ACCESS_KEY_ID"] = idlSession['credentials']['accessKeyId']
os.environ["AWS_SECRET_ACCESS_KEY"] = idlSession['credentials']['secretAccessKey']
os.environ["AWS_SESSION_TOKEN"] = idlSession['credentials']['sessionToken']

   # Now operations against the IDL S3 bucket can be performed directly
# ...

# upgrade pip and install required libraries


HEADERS = {
'Accept': '*/*',
'Accept-Encoding': 'gzip, deflate, br',
'Connection': 'keep-alive',
'Content-Type': 'application/json'
}

# The base gateway that provides transparent authentication
GATEWAY = os.environ['GATEWAY_ENDPOINT'] + '/gateway/'

# Get a signed URL for down/upload of data. The function
# attempts for 5 times to obtain the URL and then raises an exception.
# The same function can also generate an upload signed url
def getSignedURL(fileName, folder, attempt=0, upload=True): 
  if upload:
    IDLpath = 'datalake/v3/generateUploadObjectUrls'
  else:
    IDLpath = 'datalake/v3/generateDownloadObjectUrls'

  IDLFilePath = '/%s/%s' % (folder, fileName)
  url = GATEWAY + IDLpath
  body='{"paths": [{"path": "%s"}]}' % IDLFilePath

  # Send request to receive a signed url
  response = requests.post(url, headers=HEADERS, data=body)

  try:
    # Return the signed url
    return json.loads(response.text)['objectUrls'][0]['signedUrl']
  except KeyError:
    if attempt < 5:
        attempt += 1
        return getSignedURL(fileName, attempt, upload)
    else:
        raise Exception('Failed to get a signed URL')


def write_to_idl(fileName, folder):
    signedURL = getSignedURL(fileName, folder)
    # upload the test file using the signed url
    with open(fileName, "rb") as f:
        requests.put(signedURL, headers=HEADERS, data=f)
    
    if response.status_code == 200 or response.status_code == 201:
        print("✅ File uploaded successfully!")
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")

# 示例
# OUTPUT_FOLDER = 'cnc_pdm'
# fileName = "pcba_heater_hs_predict_wide.csv"
# write_to_idl(filename, OUTPUT_FOLDER)
```
:::

::: {#8917f389-cd35-40e4-a4fb-ed163ea6e985 .cell .code}
``` python
# 从IDL下载文件

def generateIDLAccessToken(permission: str):
    dlpath = '/datalake/v3/generateAccessToken'
    gw = os.environ['GATEWAY_ENDPOINT'] + '/gateway/'

    # increment_value = 1
    headers = {
        'Content-Type': 'application/json'
    }

    payload = json.dumps({
        "path": "/",
        "permission": permission
    })

    #print(payload)

    dl_url = gw + dlpath

    response = requests.post(dl_url, data=payload, headers=headers)
    print(response.status_code)
    #print(response.headers)
    #print(response.text)

    dl = json.loads(response.text)
    print(dl)
    os.environ["AWS_ACCESS_KEY_ID"] = dl['credentials']['accessKeyId']
    os.environ["AWS_SECRET_ACCESS_KEY"] = dl['credentials']['secretAccessKey']
    os.environ["AWS_SESSION_TOKEN"] = dl['credentials']['sessionToken']
    return 's3://' + dl['storageAccount'] + '/' +  dl['storagePath']

idl_base= generateIDLAccessToken("READ")
print(idl_base)

# 从IDL下载文件

# 使用 AWS CLI 或 boto3 从数据湖下载模型文件[6,7](@ref)
    # 这里以使用 boto3 为例

import os
def download_idl_data(obj_path, model_filename):
    try:
        # 3. 使用 awscli 下载模型文件 (假设你知道模型在数据湖中的完整路径)
        # 例如，假设模型文件在数据湖中的完整路径是: s3://datalake-prod-a-presiot-1586345024501/data/ten=presiot/LXDData/CNC_Model/best_cnc_vibration_model.pkl
        model_s3_path = f"s3://datalake-prod-a-presiot-1586345024501/data/ten=presiot/{obj_path}"
       # scaler_s3_path = "s3://datalake-prod-a-presiot-1586345024501/data/ten=presiot/LXDData/CNC_Model/scaler.pkl"  # 标准化器路径

        # 使用 aws s3 cp 命令下载
        os.system(f"aws s3 cp {model_s3_path} {model_filename}")
       # os.system(f"aws s3 cp {scaler_s3_path} {scaler_filename}")
        print("模型文件已从数据湖下载并保存到本地。")

    except Exception as e:
        print(f"从数据湖下载模型文件失败: {e}")
        # 可以考虑在这里添加训练模型并保存的逻辑
        raise

# 示例
# download_idl_data("zhulan/qpdatasample_v2.csv", "mydata.csv")
```
:::

::: {#08a27264-1515-4d19-83c8-1bc4f35c4a7b .cell .code}
``` python
download_idl_data("cnc_pdm/input/CNC/data/predictive_maintenance.csv", "predictive_maintenance.csv")
```
:::

::: {#9a69a924 .cell .markdown papermill="{\"duration\":1.4897e-2,\"end_time\":\"2023-11-04T11:27:31.955594\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:31.940697\",\"status\":\"completed\"}" tags="[]"}
## **引言**

对于工业4.0背景下的企业而言，亟需一种能够判断特定设备是否会发生故障以及故障类型的方法，这一点至关重要。背后的核心原因在于以下考量：维修或更换一台发生故障的整机，其所需成本通常远高于更换单个零部件的成本。因此，安装用于监测设备状态并收集相关信息的传感器，能够为企业大幅节省开支。

本文使用来自加州大学欧文分校机器学习知识库（UCI
Repository）的AI4I预测性维护数据集（AI4I Predictive Maintenance
Dataset）开展分析。内容包括：

1.  对数据集进行探索性分析，以深入了解数据特征；
2.  应用多种数据预处理技术，为后续用于预测的算法准备数据；
3.  本模型主要涉及两项任务，一是判断某一通用设备是否即将发生故障，二是确定故障的具体类型；
4.  对任务所得结果进行对比分析，既通过适当的指标评估各模型的性能，也对其可解释性展开探讨。解释性。
:::

::: {#41678bec .cell .markdown jp-MarkdownHeadingCollapsed="true" papermill="{\"duration\":1.4156e-2,\"end_time\":\"2023-11-04T11:27:31.984350\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:31.970194\",\"status\":\"completed\"}" tags="[]"}
## 目录

1.  [任务和数据描述](#description)
2.  [数据探索分析](#EDA) `<br>`{=html} 2.1 [去除ID列](#ID) `<br>`{=html}
    2.2 [Target anomalies](#target) `<br>`{=html} 2.3
    [离群检测](#outliers) `<br>`{=html} 2.4
    [使用SMOTE重采样](#resampling) `<br>`{=html} 2.5
    [重采样置换的对比](#resample_comparison) `<br>`{=html} 2.6
    [特征缩放和编码](#encoding) `<br>`{=html} 2.7
    [PCA主成分分析和相关性矩阵](#pca) `<br>`{=html} 2.8
    [模型评估](#metrics) `<br>`{=html}
3.  [二元分类](#binary) `<br>`{=html} 3.1
    [任务解释和定义](#preliminaries) `<br>`{=html} 3.2
    [特征选择和探索](#selection) `<br>`{=html} 3.3 [逻辑回归
    Benchmark](#binary_benchmark) `<br>`{=html} 3.4
    [多模型拟合](#binary_models) `<br>`{=html}
4.  [多元分类](#multi) `<br>`{=html} 4.1 [逻辑回归
    Benchmark](#multi_benchmark) `<br>`{=html} 4.2
    [多模型拟合](#multi_models) `<br>`{=html}
5.  [决策树路径可视化](#decisionpath) `<br>`{=html}
6.  [结论](#conclusions) `<br>`{=html}
7.  [保存模型](#savemodel) `<br>`{=html}
:::

::: {#fadeffca .cell .markdown jp-MarkdownHeadingCollapsed="true" papermill="{\"duration\":1.5469e-2,\"end_time\":\"2023-11-04T11:27:32.014482\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:31.999013\",\"status\":\"completed\"}" tags="[]"}
## 1) **任务和数据描述** `<a id="description">`{=html}`</a>`{=html} {#1-任务和数据描述-}

从工业场景下采集的典型设备数据进行预测性维护。

该数据集包含10,000条记录，每条记录有14个特征：

UDI,Product ID,Type,Air temperature \[K\],Process temperature
\[K\],Rotational speed \[rpm\],Torque \[Nm\],Tool wear
\[min\],Target,Failure Type

-   UDI：唯一标识符，范围从1到10,000；
-   产品ID：由一个字母（L、M 或
    H）表示产品质量等级（低质量占60%，中等占30%，高质量占10%），后跟一个特定序列号；
-   质量类型(Type)：L、M、H
-   空气温度\[K\]：通过随机游走过程生成，后标准化至围绕300 K、标准差为2
    K的分布；（公式： °C = K - 273.15）​
-   工艺温度 \[K\]：通过随机游走过程生成，标准化至标准差为1
    K，再叠加于空气温度之上加10 K；
-   转速 \[rpm\]：基于2860
    W功率计算得出，并叠加正态分布噪声；（）三相异步电动机（鼠笼式电机）​​
    在工频电网驱动下的转速。​​功率 (P) = 扭矩 (τ) × 角速度 (ω)​
-   扭矩 \[Nm\]：扭矩值围绕40 Nm呈正态分布，标准差为10 Nm，无负值；
-   刀具磨损（Tool wear）
    \[min\]：质量等级H/M/L分别在工艺过程中增加5/3/2分钟的刀具磨损量；（根据质量LMH算出来的）
-   机器故障：标签，指示在该数据点上机器是否因以下任一故障模式而失效。机器故障由五种独立故障模式构成：
    -   刀具磨损失效(tool wear failure )
        (TWF)：刀具将在200--240分钟内随机选定的时间点被更换或失效；
    -   散热失效(heat dissipation failure)
        (HDF)：当空气温度与工艺温度之差低于8.6 K，且刀具转速低于1380
        rpm时，散热导致工艺失败；
    -   功率失效（power failure）
        (PWF)：扭矩与转速（换算为弧度/秒）的乘积等于工艺所需功率。若此功率低于3500
        W或高于9000 W，则工艺失败；
    -   过载失效 (overstrain failure)
        (OSF)：当刀具磨损量与扭矩的乘积超过特定阈值（L型产品为11,000
        minNm，M型为12,000，H型为13,000）时，因过载导致工艺失败；
    -   随机失效(random failures )
        (RNF)：每个工艺过程均有0.1%的概率独立于任何参数发生失效。

只要上述任一故障模式成立，机器故障标签(Column=Target)即设为1。
:::

::: {#921c6223 .cell .markdown papermill="{\"duration\":1.4442e-2,\"end_time\":\"2023-11-04T11:27:32.043450\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:32.029008\",\"status\":\"completed\"}" tags="[]"}
## 2) **数据探索分析** `<a id="EDA">`{=html}`</a>`{=html} {#2-数据探索分析-}

我们的数据探索首先检查每个条目是否唯一，是否有重复项。这是通过确认唯一ProductID的编号来实现的，然后，打印报告查找缺失值，并检查每个列的数据类型。
:::

::: {#096a50b4 .cell .code papermill="{\"duration\":1.304057,\"end_time\":\"2023-11-04T11:27:33.361940\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:32.057883\",\"status\":\"completed\"}" tags="[]"}
``` python

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入数据
data_path = 'predictive_maintenance.csv'
data = pd.read_csv(data_path)
n = data.shape[0]

# 检查空值和数据类型
print('特征非空值和数据类型:')
data.info()
print('检查重复值:', data['Product ID'].unique().shape[0]!=n)
```
:::

::: {#1aa9498c .cell .code papermill="{\"duration\":2.7527e-2,\"end_time\":\"2023-11-04T11:27:33.433413\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:33.405886\",\"status\":\"completed\"}" tags="[]"}
``` python
# 将数字列设置为浮点数类型
data['Tool wear [min]'] = data['Tool wear [min]'].astype('float64')
data['Rotational speed [rpm]'] = data['Rotational speed [rpm]'].astype('float64')

# 特征重命名
data.rename(mapper={'Air temperature [K]': 'Air temperature',
                    'Process temperature [K]': 'Process temperature',
                    'Rotational speed [rpm]': 'Rotational speed',
                    'Torque [Nm]': 'Torque',
                    'Tool wear [min]': 'Tool wear'}, axis=1, inplace=True)

```
:::

::: {#39c61c36 .cell .markdown papermill="{\"duration\":1.4892e-2,\"end_time\":\"2023-11-04T11:27:33.462698\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:33.447806\",\"status\":\"completed\"}" tags="[]"}
### 2.1) 去除ID列 `<a id="ID">`{=html}`</a>`{=html} {#21-去除id列-}

机器故障与其标识符之间本质上是不存在关联的，因此可以合理将其删除。但是可以通过直方图展示不同产品ID的样分布情况。
:::

::: {#cf7aa8e4 .cell .code papermill="{\"duration\":0.353843,\"end_time\":\"2023-11-04T11:27:33.831662\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:33.477819\",\"status\":\"completed\"}" tags="[]"}
``` python

# 去掉ProductID的首字母LMH，并把剩余的数字字符换为数值类型
data['Product ID'] = data['Product ID'].apply(lambda x: x[1:])
data['Product ID'] = pd.to_numeric(data['Product ID'])

# 通过直方图显示去除首字母的ProductID, 并按原始文件中定义的LMH Type来分类显示
sns.histplot(data=data, x='Product ID', hue='Type')
plt.show()
```
:::

::: {#04b0e973 .cell .code papermill="{\"duration\":2.4e-2,\"end_time\":\"2023-11-04T11:27:33.870252\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:33.846252\",\"status\":\"completed\"}" tags="[]"}
``` python
# 去掉ID相关的列 UDI, Product ID
df = data.copy()
df.drop(columns=['UDI','Product ID'], inplace=True)
```
:::

::: {#4675d42d .cell .markdown papermill="{\"duration\":1.4082e-2,\"end_time\":\"2023-11-04T11:27:33.898746\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:33.884664\",\"status\":\"completed\"}" tags="[]"}
以下饼图显示了按类型划分的机器百分比：
:::

::: {#5c8bf4c6 .cell .code papermill="{\"duration\":0.110678,\"end_time\":\"2023-11-04T11:27:34.023825\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:33.913147\",\"status\":\"completed\"}" tags="[]"}
``` python
# 产品加工件类型饼图
value = data['Type'].value_counts()
Type_percentage = 100*value/data.Type.shape[0]
labels = Type_percentage.index.array
x = Type_percentage.array
plt.pie(x, labels = labels, colors=sns.color_palette('tab10')[0:3], autopct='%.0f%%')
plt.title('产品加工件类型饼图')
plt.show()
```
:::

::: {#54894457 .cell .markdown papermill="{\"duration\":1.8374e-2,\"end_time\":\"2023-11-04T11:27:34.061378\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.043004\",\"status\":\"completed\"}" tags="[]"}
### 2.2) Target anomalies `<a id="target">`{=html}`</a>`{=html} {#22-target-anomalies-}

在本节中，我们观察目标的分布，以发现任何不平衡并加以纠正
在分割数据集之前。 数据集描述的第一个异常是，当故障是随机的（RNF）时
机器故障功能未设置为1。
:::

::: {#01d8660d .cell .code papermill="{\"duration\":4.1569e-2,\"end_time\":\"2023-11-04T11:27:34.121921\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.080352\",\"status\":\"completed\"}" tags="[]"}
``` python
# 从去除了ID的DataFrame中，进一步筛选出包括Target列和Failure Type列
features = [col for col in df.columns 
            if df[col].dtype=='float64' or col =='Type']
target = ['Target','Failure Type']

# 把所有随机故障（RNF=1）的数据筛选出来
idx_RNF = df.loc[df['Failure Type']=='Random Failures'].index
df.loc[idx_RNF,target]
```
:::

::: {#29d6f47d .cell .markdown papermill="{\"duration\":1.5052e-2,\"end_time\":\"2023-11-04T11:27:34.153167\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.138115\",\"status\":\"completed\"}" tags="[]"}
幸运的是，机器故障RNF仅在18次观测中发生(小概率事件)，并且具有随机性。因此不可预测，可以从样本中删除。
:::

::: {#1b6323c2 .cell .code papermill="{\"duration\":3.1044e-2,\"end_time\":\"2023-11-04T11:27:34.199998\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.168954\",\"status\":\"completed\"}" tags="[]"}
``` python
first_drop = df.loc[idx_RNF,target].shape[0]
print('观测到的RNF=1,但是Target=0(机器没有故障)出现的次数:',first_drop)
# 把对应的这些行删除
df.drop(index=idx_RNF, inplace=True)
```
:::

::: {#e456345a .cell .markdown papermill="{\"duration\":1.5688e-2,\"end_time\":\"2023-11-04T11:27:34.230955\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.215267\",\"status\":\"completed\"}" tags="[]"}
同理，我们发现当所有类型的机器故障都设置为1(Target=1)时，Failure
Type却等于0，这些也是需要剔除的噪音。
:::

::: {#74cde1d5 .cell .code papermill="{\"duration\":3.8766e-2,\"end_time\":\"2023-11-04T11:27:34.284933\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.246167\",\"status\":\"completed\"}" tags="[]"}
``` python
# Portion of data where Machine failure=1 but no failure cause is specified
idx_ambiguous = df.loc[(df['Target']==1) &
                       (df['Failure Type']=='No Failure')].index
second_drop = df.loc[idx_ambiguous].shape[0]
print('Number of ambiguous observations:', second_drop)
display(df.loc[idx_ambiguous,target])
df.drop(index=idx_ambiguous, inplace=True)
```
:::

::: {#a8410b16 .cell .code papermill="{\"duration\":2.6543e-2,\"end_time\":\"2023-11-04T11:27:34.326857\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.300314\",\"status\":\"completed\"}" tags="[]"}
``` python
# 被剔除样本的比例
print('被剔除样本的比例:',
     (100*(first_drop+second_drop)/n))
df.reset_index(drop=True, inplace=True)   # Reset index
n = df.shape[0]
```
:::

::: {#8e016f26 .cell .markdown papermill="{\"duration\":1.531e-2,\"end_time\":\"2023-11-04T11:27:34.390661\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.375351\",\"status\":\"completed\"}" tags="[]"}
### 2.3) 离群检测 `<a id="outliers">`{=html}`</a>`{=html} {#23-离群检测-}

本节的目标是检查数据集是否包含任何离群值，这些样本点通常具有误导性。
:::

::: {#1ba55713 .cell .code papermill="{\"duration\":4.9307e-2,\"end_time\":\"2023-11-04T11:27:34.455610\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.406303\",\"status\":\"completed\"}" tags="[]"}
``` python
df.describe()
```
:::

::: {#4a3ce290 .cell .markdown papermill="{\"duration\":1.5459e-2,\"end_time\":\"2023-11-04T11:27:34.486917\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.471458\",\"status\":\"completed\"}" tags="[]"}
通过上面的数值可以初步猜测，转速和扭矩存在离群值。因为75%的样本密度值和最大值Max有较大的差异，表示出现了较大的离群。
:::

::: {#cd2de1f8 .cell .code papermill="{\"duration\":1.850124,\"end_time\":\"2023-11-04T11:27:36.352537\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:34.502413\",\"status\":\"completed\"}" tags="[]"}
``` python
num_features = [feature for feature in features if df[feature].dtype=='float64']
# 数值类型特征直方图,用于分析样本的分布
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('数值类型特征直方图')
for j, feature in enumerate(num_features):
    sns.histplot(ax=axs[j//3, j-3*(j//3)], data=df, x=feature)
plt.show()

# 数值类型箱线图，用于分析样本的离群值
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('数值类型特征箱线图')
for j, feature in enumerate(num_features):
    sns.boxplot(ax=axs[j//3, j-3*(j//3)], data=df, x=feature)
plt.show()
```
:::

::: {#ded3d2e6 .cell .markdown papermill="{\"duration\":1.6264e-2,\"end_time\":\"2023-11-04T11:27:36.385640\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:36.369376\",\"status\":\"completed\"}" tags="[]"}
通过直方图我们可以观测到，扭矩的样本呈现出非常标准的正态分布，这种情况下，使用高斯分布的3σ
规则反而比使用IQR四分位距更能更好的体现离群值。
:::

::: {#5ebca200 .cell .markdown papermill="{\"duration\":1.6179e-2,\"end_time\":\"2023-11-04T11:27:36.419489\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:36.403310\",\"status\":\"completed\"}" tags="[]"}
### 2.4) 使用SMOTE解决样本数据不平衡问题 `<a id="resampling">`{=html}`</a>`{=html} {#24-使用smote解决样本数据不平衡问题-}

另一个重要考虑因素是机器故障的发生率极低, 仅占整个数据集的3.31%。
此外，一个饼图显示每次故障所涉及的原因的发生揭示了进一步的不平衡程度。
:::

::: {#8832cd84 .cell .code papermill="{\"duration\":0.117979,\"end_time\":\"2023-11-04T11:27:36.553877\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:36.435898\",\"status\":\"completed\"}" tags="[]"}
``` python
# 故障比例可视化
idx_fail = df.loc[df['Failure Type'] != 'No Failure'].index
df_fail = df.loc[idx_fail]
df_fail_percentage = 100*df_fail['Failure Type'].value_counts()/df_fail['Failure Type'].shape[0]
print('样本中的故障比例%:', round(100*df['Target'].sum()/n,2))

plt.title('故障原因占比饼图')
plt.pie(x=df_fail_percentage.array, labels=df_fail_percentage.index.array,
        colors=sns.color_palette('tab10')[0:4], autopct='%.0f%%')
plt.show()
```
:::

::: {#59524cc3 .cell .markdown}
前面通过样本的统计分析可以得出结论：数据里97%都是"正常"，只有3%是"故障"。如果AI偷懒，永远只猜"正常"，准确率也能有97%，这就叫数据不平衡只学得会认多数派（正常机器），学不会认少数派（故障机器）。

那么如何解决这个问题呢？有三种"造数据"的方法：

1.  ​砍掉一些多数派（Under-sampling）​​：比如直接删掉很多"正常"的数据，让"正常"和"故障"的数量一样少。​缺点​是太浪费了！本来数据就不多，还主动扔掉，模型会不精准。
2.  ​复制一些少数派（Over-Sampling）​​：把现有的那点"故障"数据复制粘贴很多遍，凑够数量。​缺点​：AI会记住这几条一模一样的故障记录，反而学不会
    generalize（举一反三）。
3.  ​用SMOTE技术，智能地，​"无中生有"地造出新的、合理的、（高级Over-Sampling）​​。这是我们推荐的方法，更聪明。

​SMOTE是怎么个"智能"法？（重点）​​

1.  ​找朋友​：先随便找一个"故障"数据点（A）。
2.  ​找邻居​：找到和小A最相似的另外几个"故障"数据点（它的邻居们）。
3.  ​连线造点​：从这些邻居里随机选一个（B）。在A和B之间连一条线。
4.  ​在线随机蹦迪​：在这条线上的某个随机位置（比如中点，或者靠近A的3/4处）​凭空生成一个新的数据点。

这个新造出来的点，既像A又像B，但不是任何一个的复制品，是一个全新的、合理的"虚拟故障"数据。​这么做的巨大好处是：​​既增加了少数派（故障）的数据量，又保证了新数据不是简单的重复，和真实数据很像，能让AI更好地学习故障的特征。

​最终目标：​​通过SMOTE，把数据集调整成"正常"占80%，"故障"占20%，让AI能公平地学习两者，不再偏科。
:::

::: {#e5bb1738 .cell .markdown}
下面这段代码把\"机器正常\"的庞大数据砍到80%，同时给各类故障数据\"无中生有\"造合理的新例子，逼着AI学习到真正的内容。

​SMOTENC​：这是SMOTE的变种，专门处理混合数据​（数值型+分类型）。
:::

::: {#b0ab9ac3 .cell .code papermill="{\"duration\":0.871726,\"end_time\":\"2023-11-04T11:27:37.476684\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:36.604958\",\"status\":\"completed\"}" tags="[]"}
``` python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

# n_working 代表重采样数据占比，从之前的97%降到80%
n_working = df['Failure Type'].value_counts()['No Failure']
desired_length = round(n_working/0.8)


# 剩下20%的样本总数均分给4类故障
spc = round((desired_length-n_working)/4)  # spc - 每种类型的样本数量
balance_cause = {'No Failure':n_working,
                 'Overstrain Failure':spc,
                 'Heat Dissipation Failure':spc,
                 'Power Failure':spc,
                 'Tool Wear Failure':spc}

# 重采样
# categorical_features=[0,7]：指定第0列[Type]和第7列[Failure Type]是分类特征。
sm = SMOTENC(categorical_features=[0,7], sampling_strategy=balance_cause, random_state=0)
df_res, y_res = sm.fit_resample(df, df['Failure Type'])
```
:::

::: {#042df37d .cell .markdown papermill="{\"duration\":1.7063e-2,\"end_time\":\"2023-11-04T11:27:37.512272\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:37.495209\",\"status\":\"completed\"}" tags="[]"}
### 2.5) 重采样之后的对比 `<a id="resample_comparison">`{=html}`</a>`{=html} {#25-重采样之后的对比-}
:::

::: {#e5f202a9 .cell .code papermill="{\"duration\":0.179724,\"end_time\":\"2023-11-04T11:27:37.708464\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:37.528740\",\"status\":\"completed\"}" tags="[]"}
``` python
# 重采样后故障比例可视化
idx_fail_res = df_res.loc[df_res['Failure Type'] != 'No Failure'].index
df_res_fail = df_res.loc[idx_fail_res]
fail_res_percentage = 100*df_res_fail['Failure Type'].value_counts()/df_res_fail.shape[0]

# 百分比
print('重采样后观察值的百分比增加:', round((df_res.shape[0]-df.shape[0])*100/df.shape[0],2))
print('SMOTE重采样后故障的百分比:',  round(df_res_fail.shape[0]*100/df_res.shape[0],2))

# Pie plot
fig, axs = plt.subplots(ncols=2, figsize=(12,4))
fig.suptitle('机器故障的原因占比对比')
axs[0].pie(x=df_fail_percentage.array, labels=df_fail_percentage.index.array,
        colors=sns.color_palette('tab10')[0:4], autopct='%.0f%%')
axs[1].pie(x=fail_res_percentage.array, labels=fail_res_percentage.index.array,
        colors=sns.color_palette('tab10')[0:4], autopct='%.0f%%')
axs[0].title.set_text('原始样本')
axs[1].title.set_text('重采样后样本')
plt.show()
```
:::

::: {#25723d2e .cell .markdown}
通过KDE(Kernel Density
Estimation)核密度估计把数据点变成平滑的曲线，直观看出哪里数据多、哪里数据少。

将重采样后的样本按照不同的维度来进行KDE分析。
:::

::: {#1b4e5d28 .cell .code papermill="{\"duration\":1.192501,\"end_time\":\"2023-11-04T11:27:38.968037\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:37.775536\",\"status\":\"completed\"}" tags="[]"}
``` python
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(19,7))
fig.suptitle('重采样后的特征分布可视化分析')
custom_palette = {'L':'tab:blue', 'M':'tab:orange', 'H':'tab:green'}
for j, feature in enumerate(num_features):
    sns.kdeplot(ax=axs[j//3, j-3*(j//3)], data=df_res, x=feature,
              hue='Type', fill=True, palette=custom_palette)
plt.show()
```
:::

::: {#9745e48c .cell .markdown}
通过上面的KDE图，可以直观的看出：​低质量机器（L型）最容易坏，中质量（M型）次之，高质量（H型）很少坏。但这到底是真实现象，还是数据统计的\"假象\"？当用SMOTE技术增加故障样本数量后，这种差距更是被放大了（L型故障数据变得更多？？）。
但是通过进一步分析可以发现，除了Tool
wear刀具磨损稍微的L部分稍微有些不同（两个峰值）之外，不同质量等级的机器在其他特征上分布几乎相同。证明了L型机器故障多主要是因为样本数量多，而不是因为这些机器在技术参数上有本质差异。
:::

::: {#80f40255 .cell .markdown papermill="{\"duration\":1.9169e-2,\"end_time\":\"2023-11-04T11:27:39.006624\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:38.987455\",\"status\":\"completed\"}" tags="[]"}
接下来，以机器故障(Target)的数量，让我们看看重采样前后特征的分布是如何变化的。
:::

::: {#be9d52f2 .cell .code papermill="{\"duration\":3.363324,\"end_time\":\"2023-11-04T11:27:42.389148\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:39.025824\",\"status\":\"completed\"}" tags="[]"}
``` python
# 重采样之前
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('原始特征分布')
enumerate_features = enumerate(num_features)
for j, feature in enumerate_features:
    sns.kdeplot(ax=axs[j//3, j-3*(j//3)], data=df, x=feature,
                hue='Target', fill=True, palette='tab10')
plt.show()

# 重采样之后
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('重采样后的特征分布')
enumerate_features = enumerate(num_features)
for j, feature in enumerate_features:
    sns.kdeplot(ax=axs[j//3, j-3*(j//3)], data=df_res, x=feature,
                hue=df_res['Target'], fill=True, palette='tab10')
plt.show()

# 重采样之后 - 按故障类型下钻
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('重采样后的特征分布 - 按故障类型下钻')
enumerate_features = enumerate(num_features)
for j, feature in enumerate_features:
    sns.kdeplot(ax=axs[j//3, j-3*(j//3)], data=df_res, x=feature,
                hue=df_res['Failure Type'], fill=True, palette='tab10')
plt.show()
```
:::

::: {#5ec830d8 .cell .markdown papermill="{\"duration\":2.1897e-2,\"end_time\":\"2023-11-04T11:27:42.433521\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:42.411624\",\"status\":\"completed\"}" tags="[]"}
我们可以观察到的第一件事是，数据增强是成功完成的，因为故障实例的特征分布没有明显扭曲。

1.  ​数据增强（SMOTE）是成功的​

    -   生成的新故障数据没有扭曲原始数据的分布规律（即"造"出来的数据看起来很真实）。

2.  ​故障数据的特征规律​

    -   在转速（Rotational Speed）、扭矩（Torque）、刀具磨损（Tool
        Wear）​这三个特征上，故障数据明显集中在极端值区域（比如转速过高/过低、扭矩极大/极小）。
    -   这说明之前发现的"异常值"（outliers）​不是数据错误，而是真实存在的故障模式。

3.  ​不同故障类型的独特模式​

    -   ​转速和扭矩​：
        -   不同故障类型的数据分布几乎对称​（比如"过热故障"和"过载故障"可能分别对应转速过高和过低）。
    -   ​刀具磨损​：
        -   低值区：电力故障（PWF）和散热故障（HDF）集中在此（比如磨损轻微时易发生）。
        -   高值区：刀具磨损故障（TWF）和过劳故障（OSF）集中在此（比如磨损严重时易发生）。
    -   这些规律和数据集描述文档中的定义完全一致，进一步证明数据可信。
:::

::: {#3f6666f8 .cell .markdown papermill="{\"duration\":2.2155e-2,\"end_time\":\"2023-11-04T11:27:42.478330\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:42.456175\",\"status\":\"completed\"}" tags="[]"}
### 2.6) 特征缩放和编码 `<a id="encoding">`{=html}`</a>`{=html} {#26-特征缩放和编码-}
:::

::: {#f0291c84 .cell .code}
``` python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sc = StandardScaler()
type_dict = {'L': 0, 'M': 1, 'H': 2}
cause_dict = {'No Failure': 0,
              'Power Failure': 1,
              'Overstrain Failure': 2,
              'Heat Dissipation Failure': 3,
              'Tool Wear Failure': 4}

# 创建副本
df_pre = df_res.copy()

# 推荐：使用 map() 代替 replace()（避免警告）
df_pre['Type'] = df_pre['Type'].map(type_dict)
df_pre['Failure Type'] = df_pre['Failure Type'].map(cause_dict)

# 数值特征标准化
df_pre[num_features] = sc.fit_transform(df_pre[num_features])
```
:::

::: {#d85e989a .cell .markdown papermill="{\"duration\":2.1583e-2,\"end_time\":\"2023-11-04T11:27:42.593442\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:42.571859\",\"status\":\"completed\"}" tags="[]"}
### 2.7) PCA主成分分析和相关性矩阵 `<a id="pca">`{=html}`</a>`{=html} {#27-pca主成分分析和相关性矩阵-}

通过PCA将14维的特征降阶成3维。
:::

::: {#03a3d6fa .cell .code papermill="{\"duration\":4.9705e-2,\"end_time\":\"2023-11-04T11:27:42.664280\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:42.614575\",\"status\":\"completed\"}" tags="[]"}
``` python
pca = PCA(n_components=len(num_features))
X_pca = pd.DataFrame(data=pca.fit_transform(df_pre[num_features]), columns=['PC'+str(i+1) for i in range(len(num_features))])
var_exp = pd.Series(data=100*pca.explained_variance_ratio_, index=['PC'+str(i+1) for i in range(len(num_features))])
print('每个成分的可解释性方差比:', round(var_exp,2), sep='\n')
print('前三个成分的可解释性方差比之和: '+str(round(var_exp.values[:3].sum(),2)))
```
:::

::: {#75f1bc49 .cell .markdown papermill="{\"duration\":2.8872e-2,\"end_time\":\"2023-11-04T11:27:42.723496\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:42.694624\",\"status\":\"completed\"}" tags="[]"}
可以发现前三个分量足以几乎完全表示数据的方差，所以我们将把它们投影到三维空间中（降阶到3维）。

载荷分析就是研究主成分（或因子）与原始变量之间相关关系的过程。​它回答了一个关键问题：​​"我生成的新主成分，到底是由哪些原始变量构成的？它们的贡献权重如何？"
:::

::: {#8b0975ac .cell .code papermill="{\"duration\":0.473538,\"end_time\":\"2023-11-04T11:27:43.227038\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:42.753500\",\"status\":\"completed\"}" tags="[]"}
``` python
# 3个主成分的可视化
pca3 = PCA(n_components=3)
X_pca3 = pd.DataFrame(data=pca3.fit_transform(df_pre[num_features]), columns=['PC1','PC2','PC3'])

# 载荷分析
fig, axs = plt.subplots(ncols=3, figsize=(18,4))
fig.suptitle('载荷大小')
pca_loadings = pd.DataFrame(data=pca3.components_, columns=num_features)
for j in range(3):
    ax = axs[j]
    sns.barplot(ax=ax, x=pca_loadings.columns, y=pca_loadings.values[j])
    ax.tick_params(axis='x', rotation=90)
    ax.title.set_text('PC'+str(j+1))
plt.show()  
```
:::

::: {#30e77b82 .cell .markdown papermill="{\"duration\":2.173e-2,\"end_time\":\"2023-11-04T11:27:43.271770\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:43.250040\",\"status\":\"completed\"}" tags="[]"}
主成分权重的条形图便于理解它们代表什么：

-   PC1与两个温度数据密切相关；
-   PC2可以通过机器功率来识别，是由转速和扭矩共同决定的；
-   PC3可识别为刀具磨损。
:::

::: {#643eecb4 .cell .code papermill="{\"duration\":0.846401,\"end_time\":\"2023-11-04T11:27:44.140587\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:43.294186\",\"status\":\"completed\"}" tags="[]"}
``` python
# 重命名主成分
X_pca3.rename(mapper={'PC1':'Temperature',
                      'PC2':'Power',
                      'PC3':'Tool Wear'}, axis=1, inplace=True)

# PCA画图显示及定义
color = []
col = df_pre['Failure Type'].map({0:'tab:blue',1:'tab:orange',2:'tab:green',3:'tab:red',4:'tab:purple'})
color.append(col)
idx_w = col[col == 'tab:blue'].index
color.append(col.drop(idx_w))
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
labelTups = [('No Failure','tab:blue'),
             ('Power Failure', 'tab:orange'),
             ('Overstrain Failure','tab:green'),
             ('Heat Dissipation Failure', 'tab:red'),
             ('Tool Wear Failure','tab:purple')]

fig = plt.figure(figsize=(18,6))
fig.suptitle('数据在PCA 3D空间中的分布')
full_idx = X_pca3.index

for j, idx in enumerate([full_idx,idx_fail_res]):
    ax = fig.add_subplot(1, 2, j+1, projection='3d')

    lg = ax.scatter(X_pca3.loc[idx,'Temperature'],
                    X_pca3.loc[idx,'Power'],
                    X_pca3.loc[idx,'Tool Wear'],
                    c=color[j])
    ax.set_xlabel('$Temperature$')
    ax.set_ylabel('$Power$')
    ax.set_zlabel('$Tool Wear$')
    ax.title.set_text(str(j*'不') + '包括' +' "No Failure" 特征')
    ax.view_init(35, -10) 
    custom_lines = [plt.Line2D([],[], ls="", marker='.', 
                               mec='k', mfc=c, mew=.1, ms=20) for c in colors[j:]]
    ax.legend(custom_lines, [lt[0] for lt in labelTups[j:]], 
              loc='center left', bbox_to_anchor=(1.0, .5))
      
plt.show()
```
:::

::: {#ec378afd .cell .markdown papermill="{\"duration\":2.3966e-2,\"end_time\":\"2023-11-04T11:27:44.190740\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:44.166774\",\"status\":\"completed\"}" tags="[]"}
由这三个轴生成的空间投影突出表明：

-   TWF(tool wear
    failure)是与所有其他故障最好分离的一类故障，似乎几乎完全取决于PC3（工具磨损）；
-   PWF(power
    failure)沿PC2(Power)占据两个极端波段，它独立于其他两个分量；
-   OSF(overstrain failure)和HDF(heat dissipation
    failure)类的分离程度比其他类要小，即使可以观察到第一类的特点是高刀具磨损和低功率，而第二类的特点是高温和低功率。
:::

::: {#c43b588d .cell .code papermill="{\"duration\":0.367823,\"end_time\":\"2023-11-04T11:27:44.583466\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:44.215643\",\"status\":\"completed\"}" tags="[]"}
``` python
# 特征关联矩阵
plt.figure(figsize=(7,4))
sns.heatmap(data=df_pre.corr(), mask=np.triu(df_pre.corr()), annot=True, cmap='BrBG')
plt.title('特征关联矩阵')
plt.show()
```
:::

::: {#88797ea2 .cell .markdown papermill="{\"duration\":2.4794e-2,\"end_time\":\"2023-11-04T11:27:44.635205\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:44.610411\",\"status\":\"completed\"}" tags="[]"}
不出所料，我们观察到与温度和功率(转速和扭矩)相关的特征是强相关的。

此外，Tool wear(刀具磨损)与Target(是否故障)和Failture
Type(故障类型)都是强相关性，证实了我们通过研究PCA所观察到的结果。

此外，Torque(扭矩)和Target(是否故障)和Failture
Type(故障类型)之间也观察到较弱的相关性。
:::

::: {#c69f8011 .cell .markdown papermill="{\"duration\":2.4211e-2,\"end_time\":\"2023-11-04T11:27:44.684115\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:44.659904\",\"status\":\"completed\"}" tags="[]"}
### 2.8) 模型评估 `<a id="metrics">`{=html}`</a>`{=html} {#28-模型评估-}

-   准确性：表示正确分类的实例的比例，它是分类任务中通常使用的最直观的度量。
    $$ Accuracy = \frac{TP + TN}{TP + TN + FT + FN} $$

-   AUC/ROC：可被视为衡量真阳性和真阴性之间分离的指标，即模型区分类别的能力。
    具体而言，它表示ROC曲线下方的面积，由真阳性率（召回率）的每个可能值的估计值给出。

-   F1：向Precision(精度)和Recall(召回率)报告模型的分类能力，赋予两者相同的权重。
    $$F1 = 2\frac{Precision * Recall}{Precision + Recall}$$

虽然AUC通常是有效的，但在类高度不平衡的情况下（如在二元任务中），AUC可能是乐观的，而F1分数在这种情况下更可靠。我们认为最后一个指标特别重要，因为它能够调解即将发生故障的机器被归类为功能正常的机器（召回）和功能正常的机器被归类为即将发生故障的机器（精度）的情况。更具体地说，我们将通过β参数评估F1的"调整"版本，从而赋予召回率比精度更重要：
$$F_\beta = (1 + \beta^2)\frac{Precision * Recall}{\beta^2  Precision + Recall}$$
选择β=2（文献中常见）时，召回率的影响更大。这种选择的动机是为了优化机器的维护成本，限制购买不必要的替换材料是一件好事，但避免机器损坏后不得不更换的可能性更为重要，因为第二种情况通常成本更高。
:::

::: {#1bc5d65e .cell .markdown papermill="{\"duration\":2.4784e-2,\"end_time\":\"2023-11-04T11:27:44.735849\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:44.711065\",\"status\":\"completed\"}" tags="[]"}
## 3) **二元分类** `<a id="binary">`{=html}`</a>`{=html} {#3-二元分类-}
:::

::: {#6b0db4c9 .cell .markdown papermill="{\"duration\":2.4624e-2,\"end_time\":\"2023-11-04T11:27:44.785519\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:44.760895\",\"status\":\"completed\"}" tags="[]"}
### 3.1) 任务解释和定义 `<a id="preliminaries">`{=html}`</a>`{=html} {#31-任务解释和定义-}

本节的目标是找到对数据集进行二元分类的最佳模型，以预测是否会出现机器故障。

在这个项目中，我们使用比率（80%训练集/10%验证集/10%测试集）进行数据集分割。

我们选择多种分类算法来分别拟合模型，并找到最佳的模型和参数组合：

-   逻辑回归：它估计因变量作为函数的概率独立变量。
-   K近邻（K-NN）：基于计算数据集元素之间距离的算法。如果数据与同一类的其他数据足够接近，则将其分配给某个类。参数K表示在分配类时考虑的相邻数据的数量。
-   支持向量机：其目的是在N维空间（N------特征数量）中找到一个超平面，该超平面在最大化边缘距离的同时对数据点进行清晰分类，即两类数据点之间的距离。
-   随机森林：它使用集成学习，这是一种结合许多分类器为复杂问题提供解决方案的技术。随机森林使用套袋技术：它并行构建了大量具有相同重要性的决策树，输出是大多数树选择的类。
-   梯度提升决策树（GBDT）：一种类似于随机森林的决策树集成学习算法，与随机森林的不同之处在于它使用了一种提升技术：它迭代训练一组浅决策树，每次迭代都使用前一个模型的误差残差来拟合下一个模型。最终预测是所有树预测的加权和。
-   XGBoost：是一个梯度增强决策树（GBDT）机器学习库。
:::

::: {#3ab84d13 .cell .code papermill="{\"duration\":0.167449,\"end_time\":\"2023-11-04T11:27:44.977832\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:44.810383\",\"status\":\"completed\"}" tags="[]"}
``` python
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import time

# 训练-验证-测试集划分
X, y = df_pre[features], df_pre[['Target','Failure Type']]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=df_pre['Failure Type'], random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, stratify=y_trainval['Failure Type'], random_state=0)
```
:::

::: {#2c0791fc .cell .markdown papermill="{\"duration\":2.493e-2,\"end_time\":\"2023-11-04T11:27:45.027938\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:45.003008\",\"status\":\"completed\"}" tags="[]"}
针对多算法定义一些通用的函数，用来进行模型的训练，验证和预测
:::

::: {#7ab4888e .cell .code papermill="{\"duration\":5.0114e-2,\"end_time\":\"2023-11-04T11:27:45.103953\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:45.053839\",\"status\":\"completed\"}" tags="[]"}
``` python
"""User-defined function: Evaluate cm, accurcay, AUC, F1 for a given classifier
- model, fitted estimator.
- X, data used to estimate class probabilities (paired with y_true)
- y_true, ground truth with two columns
- y_pred, predictions
- task = 'binary','multi_class'
"""
def eval_preds(model,X,y_true,y_pred,task):
    if task == 'binary':
        # Extract task target
        y_true = y_true['Target']
        cm = confusion_matrix(y_true, y_pred)
        # Probability of the minority class
        proba = model.predict_proba(X)[:,1]
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f2 = fbeta_score(y_true, y_pred, pos_label=1, beta=2)
    elif task == 'multi_class':
        y_true = y_true['Failure Type']
        cm = confusion_matrix(y_true, y_pred)
        proba = model.predict_proba(X)
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba, multi_class='ovr', average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    metrics = pd.Series(data={'ACC':acc, 'AUC':auc, 'F1':f1, 'F2':f2})
    metrics = round(metrics,3)
    return cm, metrics



"""User-defined function: Fits one estimator using GridSearch to search for the best parameters
- clf, estimator
- X, y = X_train, y_train
- params, parameters grid for GridSearch
- task = 'binary','multi_class'
"""
def tune_and_fit(clf,X,y,params,task):
    if task=='binary':
        f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,
                                cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Target'])
    elif task=='multi_class':
        f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,
                              cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Failure Type'])
        
    print('最佳参数:', grid_model.best_params_)
    # Print training times
    train_time = time.time()-start_time
    mins = int(train_time//60)
    print('训练时间: '+str(mins)+'m '+str(round(train_time-mins*60))+'s')
    return grid_model



"""User-defined function: Makes predictions using the tuned classifiers.
Then uses eval_preds to compute the relative metrics. Returns:
- y_pred, DataFrame containing the predictions of each model
- cm_list, confusion matrix list
- metrics, DataFrame containing the metrics
Input:
- fitted_models, fitted estimators
- X, data used to make predictions
- y_true, true values for target
- clf_str, list containing estimators names
- task = 'binary','multi_class'
"""
def predict_and_evaluate(fitted_models,X,y_true,clf_str,task):
    cm_dict = {key: np.nan for key in clf_str}
    metrics = pd.DataFrame(columns=clf_str)
    y_pred = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models,clf_str):
        # Update predictions
        y_pred[model_name] = fit_model.predict(X)
        # Metrics
        if task == 'binary':
            cm, scores = eval_preds(fit_model,X,y_true,
                                     y_pred[model_name],task)
        elif task == 'multi_class':
            cm, scores = eval_preds(fit_model,X,y_true,
                                     y_pred[model_name],task)
        # Update Confusion matrix and metrics
        cm_dict[model_name] = cm
        metrics[model_name] = scores
    return y_pred, cm_dict, metrics



"""User-defined function: Fit the estimators on multiple classifiers
- clf, estimators
- clf_str, list containing estimators names
- X_train,y_train, data used to fit models
- X_val,y_val, data used to validate models
"""

def fit_models(clf,clf_str,X_train,X_val,y_train,y_val):
    metrics = pd.DataFrame(columns=clf_str)
    for model, model_name in zip(clf, clf_str):
        model.fit(X_train,y_train['Target'])
        y_val_pred = model.predict(X_val)
        metrics[model_name] = eval_preds(model,X_val,y_val,y_val_pred,'binary')[1]
    return metrics
```
:::

::: {#ee2b3a22 .cell .markdown papermill="{\"duration\":2.4683e-2,\"end_time\":\"2023-11-04T11:27:45.154165\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:45.129482\",\"status\":\"completed\"}" tags="[]"}
### 3.2) 特征选择和探索 `<a id="selection">`{=html}`</a>`{=html} {#32-特征选择和探索-}

我们在训练模型之前，还可以进一步处理一下数据特征，以达到最好的效果。通过前面的特征相关性矩阵，我们观察到：

-   ​"过程温度"和"空气温度"​​是正相关的：一个高另一个也高。
-   ​​"扭矩"和"转速"​是负相关的​：一个高另一个就低。

因此我们可以考虑：

1.  把"过程温度"和"空气温度"合并成一个新特征，就叫"温差"。
2.  把"扭矩"和"转速"也合并成一个新特征，就叫"扭矩转速乘积"。

这样既解决了它们（相关性高）可能带来的问题，生成的新特征又更有实际意义，更能直接对应到故障原因上。
:::

::: {#5a22f67a .cell .code papermill="{\"duration\":30.882664,\"end_time\":\"2023-11-04T11:28:16.063678\",\"exception\":false,\"start_time\":\"2023-11-04T11:27:45.181014\",\"status\":\"completed\"}" tags="[]"}
``` python
# 构建模型
lr = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC(probability=True)
rfc = RandomForestClassifier()
xgb = XGBClassifier() 

clf = [lr,knn,svc,rfc,xgb]
clf_str = ['LR','KNN','SVC','RFC','XGB'] 

# 按原始训练集拟合
metrics_0 = fit_models(clf,clf_str,X_train,X_val,y_train,y_val)

# 通过构建Process temperature和Air temperature的乘积作为一个新的特征来进行拟合
XX_train = X_train.drop(columns=['Process temperature','Air temperature'])
XX_val = X_val.drop(columns=['Process temperature','Air temperature'])
XX_train['Temperature']= X_train['Process temperature']*X_train['Air temperature']
XX_val['Temperature'] = X_val['Process temperature']*X_val['Air temperature']
metrics_1 = fit_models(clf,clf_str,XX_train,XX_val,y_train,y_val)

# 通过构建Rotational speed和Torque的乘积作为一个新的特征来进行拟合
XX_train = X_train.drop(columns=['Rotational speed','Torque'])
XX_val = X_val.drop(columns=['Rotational speed','Torque'])
XX_train['Power'] = X_train['Rotational speed']*X_train['Torque']
XX_val['Power'] = X_val['Rotational speed']*X_val['Torque']     
metrics_2 = fit_models(clf,clf_str,XX_train,XX_val,y_train,y_val)

# 通过同时构建上述两个新的特征来进行拟合
XX_train = X_train.drop(columns=['Process temperature','Air temperature','Rotational speed','Torque'])
XX_val = X_val.drop(columns=['Process temperature','Air temperature','Rotational speed','Torque'])
XX_train['Temperature']= X_train['Process temperature']*X_train['Air temperature']
XX_val['Temperature']= X_val['Process temperature']*X_val['Air temperature']
XX_train['Power'] = X_train['Rotational speed']*X_train['Torque']
XX_val['Power'] = X_val['Rotational speed']*X_val['Torque']       
metrics_3 = fit_models(clf,clf_str,XX_train,XX_val,y_train,y_val)

# 分类指标可视化
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,8))
fig.suptitle('分类指标')
for j, model in enumerate(clf_str):
    ax = axs[j//3,j-3*(j//3)]
    model_metrics = pd.DataFrame(data=[metrics_0[model],metrics_1[model],metrics_2[model],metrics_3[model]])
    model_metrics.index = ['原始训练集','“新”温度','“新”功率','“新”温度和功率']
    model_metrics.transpose().plot(ax=ax, kind='bar', rot=0, )
    ax.title.set_text(model)
    ax.get_legend().remove()
fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
axs.flatten()[-2].legend(title='Dataset', loc='upper center',  bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12)
plt.show()
```
:::

::: {#1efb9472 .cell .markdown papermill="{\"duration\":2.4318e-2,\"end_time\":\"2023-11-04T11:28:16.113009\",\"exception\":false,\"start_time\":\"2023-11-04T11:28:16.088691\",\"status\":\"completed\"}" tags="[]"}
从上面的图中可以发现，基于原始数据集来拟合各类模型，所获得的模型评估分数都比通过减少或者创建新特征的方式要更高。因此直接考虑使用原始数据集，不再考虑特征选择或者创建了。
:::

::: {#fe6ef15f .cell .markdown papermill="{\"duration\":2.4344e-2,\"end_time\":\"2023-11-04T11:28:16.162156\",\"exception\":false,\"start_time\":\"2023-11-04T11:28:16.137812\",\"status\":\"completed\"}" tags="[]"}
### 3.3) 逻辑回归 Benchmark `<a id="binary_benchmark">`{=html}`</a>`{=html} {#33-逻辑回归-benchmark-}
:::

::: {#a02d65ac .cell .code papermill="{\"duration\":0.288971,\"end_time\":\"2023-11-04T11:28:16.476314\",\"exception\":false,\"start_time\":\"2023-11-04T11:28:16.187343\",\"status\":\"completed\"}" tags="[]"}
``` python
# Make predictions
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train['Target'])
y_val_lr = lr.predict(X_val)
y_test_lr = lr.predict(X_test)

# Metrics
cm_val_lr, metrics_val_lr = eval_preds(lr,X_val,y_val,y_val_lr,'binary')
cm_test_lr, metrics_test_lr = eval_preds(lr,X_test,y_test,y_test_lr,'binary')
print('验证集性能:',metrics_val_lr, sep='\n')
print('测试集性能:',metrics_test_lr, sep='\n')

cm_labels = ['Not Failure', 'Failure']
cm_lr = [cm_val_lr, cm_test_lr]

# 混淆矩阵
fig, axs = plt.subplots(ncols=2, figsize=(8,4))
fig.suptitle('逻辑回归结果混淆矩阵')
for j, title in enumerate(['验证集', '测试集']):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_lr[j], annot=True,
              fmt='d', cmap='Blues', cbar=False)
    axs[j].title.set_text(title)
    axs[j].set_xticklabels(cm_labels)
    axs[j].set_yticklabels(cm_labels)
plt.show()

# 逻辑回归的特征重要性
d = {'features': X_train.columns, 'odds': np.exp(lr.coef_[0])}
odds_df = pd.DataFrame(data=d).sort_values(by='odds', ascending=False)
odds_df
```
:::

::: {#f6b2f572 .cell .markdown papermill="{\"duration\":2.5151e-2,\"end_time\":\"2023-11-04T11:28:16.526628\",\"exception\":false,\"start_time\":\"2023-11-04T11:28:16.501477\",\"status\":\"completed\"}" tags="[]"}
可以发现"扭矩（Torque）"和"转速（Rotational
Speed）"这两个特征的优势太高谱，被模型赋予了"不切实际的重要性"。主要是因为这两个指标本身的波动性（variance）非常大，尤其是在机器发生故障的那些数据里，这种剧烈的波动"带偏"了模型，让模型误以为它们极度重要。通过模型的性能平方也可以看出来，针对当前的数据集特点，逻辑回归不是一个很好的选择。
:::

::: {#193de582 .cell .markdown papermill="{\"duration\":2.476e-2,\"end_time\":\"2023-11-04T11:28:16.577438\",\"exception\":false,\"start_time\":\"2023-11-04T11:28:16.552678\",\"status\":\"completed\"}" tags="[]"}
### 3.4) 多模型拟合 `<a id="binary_models">`{=html}`</a>`{=html} {#34-多模型拟合-}
:::

::: {#beec1508 .cell .code papermill="{\"duration\":413.076728,\"end_time\":\"2023-11-04T11:35:09.679069\",\"exception\":false,\"start_time\":\"2023-11-04T11:28:16.602341\",\"status\":\"completed\"}" tags="[]"}
``` python
# 多模型定义
knn = KNeighborsClassifier()
svc = SVC()
rfc = RandomForestClassifier()
xgb = XGBClassifier() 
clf = [knn,svc,rfc,xgb]
clf_str = ['KNN','SVC','RFC','XGB']

# GridSearch的初始参数
knn_params = {'n_neighbors':[1,3,5,8,10]}
svc_params = {'C': [1, 10, 100],
              'gamma': [0.1,1],
              'kernel': ['rbf'],
              'probability':[True],
              'random_state':[0]}
rfc_params = {'n_estimators':[100,300,500,700],
              'max_depth':[5,7,10],
              'random_state':[0]}
xgb_params = {'n_estimators':[300,500,700],
              'max_depth':[5,7],
              'learning_rate':[0.01,0.1],
              'objective':['binary:logistic']}
params = pd.Series(data=[knn_params,svc_params,rfc_params,xgb_params], index=clf)

# Tune hyperparameters with GridSearch (estimated time 8m)
print('开始超参数调优')
fitted_models_binary = []
for model, model_name in zip(clf, clf_str):
    print(str(model_name) + '模型超参数调... ')
    fit_model = tune_and_fit(model,X_train,y_train,params[model],'binary')
    fitted_models_binary.append(fit_model)
```
:::

::: {#659ae6f9 .cell .code papermill="{\"duration\":1.987944,\"end_time\":\"2023-11-04T11:35:11.693128\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:09.705184\",\"status\":\"completed\"}" tags="[]"}
``` python
# Create evaluation metrics
task = 'binary'
y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(fitted_models_binary,X_val,y_val,clf_str,task)
y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(fitted_models_binary,X_test,y_test,clf_str,task)

# Show Validation Confusion Matrices
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('验证集混淆矩阵')
for j, model_name in enumerate(clf_str):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_dict_val[model_name], annot=True,
                fmt='d', cmap='Blues', cbar=False)
    ax.title.set_text(model_name)
    ax.set_xticklabels(cm_labels)
    ax.set_yticklabels(cm_labels)
plt.show()

# Show Test Confusion Matrices
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('测试集混淆矩阵')
for j, model_name in enumerate(clf_str):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_dict_test[model_name], annot=True,
                fmt='d', cmap='Blues', cbar=False)
    ax.title.set_text(model_name)
    ax.set_xticklabels(cm_labels)
    ax.set_yticklabels(cm_labels)
plt.show()

# Print scores
print('')
print('验证集评分:', metrics_val, sep='\n')
print('测试集评分:', metrics_test, sep='\n')
```
:::

::: {#a525e9e1 .cell .markdown papermill="{\"duration\":2.5882e-2,\"end_time\":\"2023-11-04T11:35:11.746793\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:11.720911\",\"status\":\"completed\"}" tags="[]"}
从前面的模型评估结果，我们可以得出如下的结论：

1.  ​模型表现排名（从好到差）​​：

    -   ​XGBoost (XGB)​​：表现最好；
    -   ​随机森林 (RFC)​​ 和 ​支持向量机 (SVC)​​：表现极接近，并列第二；
    -   K-近邻 (KNN)​​：表现明显最差。

2.  所有模型在测试集上都没出现性能大幅下降，说明没有过拟合，模型泛化能力很好。

3.  XGBoost凭借最好的预测性能成为最佳模型，但其'黑盒'特性使得我们需借助其他技术（如特征重要性图）来理解其决策过程。而随机森林在表现优异的同时，提供了更好的可解释性，这体现了机器学习中准确性与可解释性之间常见的权衡。​

    -   ​随机森林 (RFC)​​
        的最佳参数是"少而深"（用较少的学习器，但每棵决策树很深）；
    -   XGBoost (XGB)​​
        的最佳参数是"多而浅"（用很多的学习器，但每棵树很浅）。
:::

::: {#bb5d666d .cell .markdown}
为了弥补XGBoost模型可解释性差的缺点，我们可以使用
​​"置换特征重要性"(Permutation Feature Importance)​​
的方法来画图展示哪个特征对模型预测的影响最大，从而让我们对模型的工作原理有大致了解。
:::

::: {#bb42f9cd .cell .code papermill="{\"duration\":34.47135,\"end_time\":\"2023-11-04T11:35:46.247953\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:11.776603\",\"status\":\"completed\"}" tags="[]"}
``` python
# Evaluate Permutation Feature Importances
f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
importances = pd.DataFrame()
for clf in fitted_models_binary:
    result = permutation_importance(clf, X_train,y_train['Target'], scoring=f2_scorer,random_state=0)
    result_mean = pd.Series(data=result.importances_mean, index=X.columns)
    importances = pd.concat(objs=[importances,result_mean],axis=1)
importances.columns = clf_str

# Barplot of Feature Importances
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('置换特征重要性')
for j, name in enumerate(importances.columns):
    sns.barplot(ax=axs[j], x=importances.index, y=importances[name].values)
    axs[j].tick_params('x',labelrotation=90)
    axs[j].set_ylabel('重要性')
    axs[j].title.set_text(str(name))
plt.show()
```
:::

::: {#2c9b6bf7 .cell .markdown papermill="{\"duration\":2.7215e-2,\"end_time\":\"2023-11-04T11:35:46.301923\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:46.274708\",\"status\":\"completed\"}" tags="[]"}
置换特征重要性是一种破坏性测试。它通过随机打乱某个特征在测试集上的数据，观察模型性能（如准确率）因此下降的幅度来评估其重要性。下降越多，说明该特征越关键。其优势在于：

-   ​结果可靠​：在独立的测试集上评估，更能反映特征的真实、泛化的影响力，避免过拟合带来的偏差。
-   ​公平可比​：提供了一个统一标准，使得不同模型（如图中的KNN、SVC、RFC、XGB）的特征重要性可以放在一起直接对比。
:::

::: {#aa76b0ae .cell .markdown papermill="{\"duration\":2.6818e-2,\"end_time\":\"2023-11-04T11:35:46.355736\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:46.328918\",\"status\":\"completed\"}" tags="[]"}
## 4) **多元任务** `<a id="multi">`{=html}`</a>`{=html} {#4-多元任务-}

接下来我们进行多分类任务，不仅预测是否会出现故障，而且预测将发生故障的类型。

对于多类目标，在计算AUC、F1和F2分数的值时，需要设置参数"average"。我们选择"平均=加权"，是为了考虑到类的不平衡：事实上，在数据预处理结束时，我们有80%的工作机器和20%的故障机器。

我们决定使用"One vs
Rest"方法，它涉及到每个类训练一个分类器，该类的样本作为正样本，所有其他样本作为负样本。
:::

::: {#74e799c1 .cell .markdown papermill="{\"duration\":2.6328e-2,\"end_time\":\"2023-11-04T11:35:46.409463\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:46.383135\",\"status\":\"completed\"}" tags="[]"}
### 4.1) 多分类逻辑回归Benchmark `<a id="multi_benchmark">`{=html}`</a>`{=html} {#41-多分类逻辑回归benchmark-}
:::

::: {#e8aef1e5 .cell .code papermill="{\"duration\":0.49874,\"end_time\":\"2023-11-04T11:35:46.934711\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:46.435971\",\"status\":\"completed\"}" tags="[]"}
``` python
# 默认 multi_class='ovr'
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train['Failure Type'])
y_val_lr = lr.predict(X_val)
y_test_lr = lr.predict(X_test)

# Validation metrics
cm_val_lr, metrics_val_lr = eval_preds(lr,X_val,y_val,y_val_lr,'multi_class')
cm_test_lr, metrics_test_lr = eval_preds(lr,X_test,y_test,y_test_lr,'multi_class')
print('验证集评估:',metrics_val_lr, sep='\n')
print('测试集评估:',metrics_test_lr, sep='\n')

cm_lr = [cm_val_lr, cm_test_lr]
cm_labels = ['No Fail','PWF','OSF','HDF','TWF']
# Show Confusion Matrices
fig, axs = plt.subplots(ncols=2, figsize=(9,4))
fig.suptitle('逻辑回归混淆矩阵')
for j, title in enumerate(['Validation Set', 'Test Set']):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_lr[j], annot=True,
              fmt='d', cmap='Blues', cbar=False)
    axs[j].title.set_text(title)
    axs[j].set_xticklabels(cm_labels)
    axs[j].set_yticklabels(cm_labels)
plt.show()

# Odds for interpretation
odds_df = pd.DataFrame(data = np.exp(lr.coef_), columns = X_train.columns, index = df_res['Failure Type'].unique())
odds_df
```
:::

::: {#98b0388a .cell .markdown papermill="{\"duration\":2.8275e-2,\"end_time\":\"2023-11-04T11:35:46.991700\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:46.963425\",\"status\":\"completed\"}" tags="[]"}
上面的结果清楚地告诉我们，​每个特征对预测某种特定故障的"贡献度"有多大。比如，某个特征的数字特别大，就说明它对这个故障的发生影响很大。

把这张表的结果和之前用另一种方法（PCA）得出的结论一对比，是完全一致的,
更加正式了PCA的主成分选择是正确的。

看表中 ​PWF​ 这种故障那一行，发现转速（Rotational Speed）​​ 和扭矩（Torque）​​
这两个特征的贡献度数字最大。这说明模型也认为PWF故障主要是它俩引起的。这正好验证了之前PCA分析时的结论：PWF故障只和第二主成分（PC2）​​
有关，而这个PC2其实就是功率​（功率 = 转速 × 扭矩）。

我们用了两种不同的分析方法，但它们都得出了相同的结论，互相印证。
:::

::: {#fd0041a8 .cell .markdown papermill="{\"duration\":2.8316e-2,\"end_time\":\"2023-11-04T11:35:47.047594\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:47.019278\",\"status\":\"completed\"}" tags="[]"}
### 4.2) 多分类的多模型拟合 `<a id="multi_models">`{=html}`</a>`{=html} {#42-多分类的多模型拟合-}
:::

::: {#9ca665ea .cell .code papermill="{\"duration\":698.177129,\"end_time\":\"2023-11-04T11:47:25.256718\",\"exception\":false,\"start_time\":\"2023-11-04T11:35:47.079589\",\"status\":\"completed\"}" tags="[]"}
``` python
# Models
knn = KNeighborsClassifier()
svc = SVC(decision_function_shape='ovr')
rfc = RandomForestClassifier()
xgb = XGBClassifier()
clf = [knn,svc,rfc,xgb]
clf_str = ['KNN','SVC','RFC','XGB']

knn_params = {'n_neighbors':[1,3,5,8,10]}
svc_params = {'C': [1, 10, 100],
              'gamma': [0.1,1],
              'kernel': ['rbf'],
              'probability':[True],
              'random_state':[0]}
rfc_params = {'n_estimators':[100,300,500,700],
              'max_depth':[5,7,10],
              'random_state':[0]}
xgb_params = {'n_estimators':[100,300,500],
              'max_depth':[5,7,10],
              'learning_rate':[0.01,0.1],
              'objective':['multi:softprob']}

params = pd.Series(data=[knn_params,svc_params,rfc_params,xgb_params], index=clf)


# Tune hyperparameters with GridSearch (estimated time 8-10m)
print('开始超参数调优')
fitted_models_multi = []
for model, model_name in zip(clf, clf_str):
    print(str(model_name) + '模型超参数调... ')
    fit_model = tune_and_fit(model,X_train,y_train,params[model],'multi_class')
    fitted_models_multi.append(fit_model)
```
:::

::: {#73e74c88 .cell .code papermill="{\"duration\":2.597821,\"end_time\":\"2023-11-04T11:47:27.881841\",\"exception\":false,\"start_time\":\"2023-11-04T11:47:25.284020\",\"status\":\"completed\"}" tags="[]"}
``` python
# Create evaluation metrics

task = 'multi_class'
y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(fitted_models_multi,X_val,y_val,clf_str,task)
y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(fitted_models_multi,X_test,y_test,clf_str,task)

# Show Validation Confusion Matrices
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('验证集混淆矩阵')
for j, model_name in enumerate(clf_str):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_dict_val[model_name], annot=True,
                fmt='d', cmap='Blues', cbar=False)
    ax.title.set_text(model_name)
    ax.set_xticklabels(cm_labels)
    ax.set_yticklabels(cm_labels)
plt.show()

# Show Test Confusion Matrices
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('测试集混淆矩阵')
for j, model_name in enumerate(clf_str):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_dict_test[model_name], annot=True, fmt='d', cmap='Blues', cbar=False)
    ax.title.set_text(model_name)
    ax.set_xticklabels(cm_labels)
    ax.set_yticklabels(cm_labels)
plt.show()

# Print scores
print('')
print('验证集评分:', metrics_val, sep='\n')
print('测试集评分:', metrics_test, sep='\n')
```
:::

::: {#a51cb217 .cell .markdown papermill="{\"duration\":2.9224e-2,\"end_time\":\"2023-11-04T11:47:27.940966\",\"exception\":false,\"start_time\":\"2023-11-04T11:47:27.911742\",\"status\":\"completed\"}" tags="[]"}
1.  ​最差：K-NN（K-近邻）​​
    -   表现最差，准确率甚至比逻辑回归还低一点。​但不必弃用，因为它训练和预测速度极快。适合用来快速初步分析，有个大概了解后再用更复杂的模型做精确分析。
2.  ​居中：SVC（支持向量机）和 RFC（随机森林）​​
    -   表现非常好且非常接近，都远超基准模型。它们的训练时间也一样快，是可靠又高效的选择。
3.  最佳：XGBoost
    -   表现最好，是精度最高的模型。​但是，它的训练时间非常长，是SVC或RFC的4倍以上。
:::

::: {#fc4c8b83 .cell .code papermill="{\"duration\":49.406658,\"end_time\":\"2023-11-04T11:48:17.377386\",\"exception\":false,\"start_time\":\"2023-11-04T11:47:27.970728\",\"status\":\"completed\"}" tags="[]"}
``` python
# Evaluate Permutation Feature Importances
f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
importances = pd.DataFrame()
for clf in fitted_models_multi:
    result = permutation_importance(clf, X_train,y_train['Failure Type'], scoring=f2_scorer,random_state=0)
    result_mean = pd.Series(data=result.importances_mean, index=X.columns)
    importances = pd.concat(objs=[importances,result_mean],axis=1)

importances.columns = clf_str

# Barplot of Feature Importances
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('置换特征重要性')
for j, name in enumerate(importances.columns):
    sns.barplot(ax=axs[j], x=importances.index, y=importances[name].values)
    axs[j].tick_params('x',labelrotation=90)
    axs[j].set_ylabel('重要性')
    axs[j].title.set_text(str(name))
plt.show()
```
:::

::: {#65a7e91c .cell .markdown papermill="{\"duration\":2.9897e-2,\"end_time\":\"2023-11-04T11:48:17.496261\",\"exception\":false,\"start_time\":\"2023-11-04T11:48:17.466364\",\"status\":\"completed\"}" tags="[]"}
## 5) **决策树路径可视化** `<a id="decisionpath">`{=html}`</a>`{=html} {#5-决策树路径可视化-}

从随机森林算法中里挑了一棵树，把这棵树的前4层决策过程画出来，让模型具备更好的可解释性。
:::

::: {#31db4da9 .cell .code papermill="{\"duration\":1.377884,\"end_time\":\"2023-11-04T11:48:18.903058\",\"exception\":false,\"start_time\":\"2023-11-04T11:48:17.525174\",\"status\":\"completed\"}" tags="[]"}
``` python
from sklearn import tree
import matplotlib.pyplot as plt

tree_binary = fitted_models_binary[2].best_estimator_.estimators_[0]
tree_multi = fitted_models_multi[2].best_estimator_.estimators_[0]
trees = [tree_binary, tree_multi]
targets = ['Target', 'Failure Type']

plt.rcParams.update({'font.size': 12})  # 全局调整字体大小

for decision_tree, target in zip(trees, targets):
    decision_tree.fit(X_train, y_train[target])
    classes = list(map(str, df_res[target].unique()))
    
    plt.figure(figsize=(25, 15))  # 增大画布尺寸
    tree.plot_tree(
        decision_tree,
        feature_names=X.columns,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=12,  # 调整节点字体大小
        max_depth=4   # 限制树的最大深度
    )
    plt.title(f"{target} Classification Tree", fontsize=14)
    plt.show()
```
:::

::: {#4db40d03 .cell .markdown jp-MarkdownHeadingCollapsed="true" papermill="{\"duration\":3.0897e-2,\"end_time\":\"2023-11-04T11:48:18.966530\",\"exception\":false,\"start_time\":\"2023-11-04T11:48:18.935633\",\"status\":\"completed\"}" tags="[]"}
## 6) **结论** `<a id="conclusions">`{=html}`</a>`{=html} {#6-结论-}

到目前为止，完成了"预测机器是否故障"和"预测故障类型"两个任务。数据分析发现，​​"温度组合"、"机器功率"（转速×扭矩）和"工具磨损"​是影响故障的最关键特征，而机器类型影响甚微。

在模型表现上，​XGBoost精度最高，但耗时较长；KNN速度最快，但精度最低。因此，最终模型选择取决于公司的实际需求：追求速度用KNN，追求精度则用XGBoost。
:::

::: {#974937c2 .cell .markdown}
## 7) **保存模型** `<a id="savemodel">`{=html}`</a>`{=html} {#7-保存模型-}
:::

::: {#04db4c58 .cell .code}
``` python
import joblib

# 保存最佳模型
model_filename = 'pdm_xgb.pkl'
joblib.dump(fitted_models_binary[3], model_filename)
print(f"模型已保存为 {model_filename}")


# 加载模型
xgb_model = joblib.load('pdm_xgb.pkl')
```
:::

::: {#3f158496-d28c-4de1-b640-e12b397f699f .cell .code}
``` python
# 保存模型到IDL

OUTPUT_FOLDER = 'cnc_pdm/output'
filename = model_filename

write_to_idl(filename, OUTPUT_FOLDER)
```
:::

::: {#dcee1b49-43c1-4955-86c9-88ac48afdfe8 .cell .code}
``` python
```
:::
