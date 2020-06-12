# COVID-19 医学论文实体抽取

## 简介



本次⽐赛聚焦于医疗（特别是和新冠肺炎相关）的知识图谱技术。数据来源包括近期发布的有关新冠肺炎的英文学术论⽂数据，并经过标注。

- 比赛共分为两个赛道。
  [赛道一任务为：医学论文实体识别](https://www.biendata.com/competition/chaindream_knowledgegraph_19_task1/)，参赛选⼿需要从中抽取出指定类型的实体（例如疾病、症状、病毒、基因、药物等）。
  [赛道二任务为：医学论文关系抽取](https://www.biendata.com/competition/chaindream_knowledgegraph_19_task2/)，参赛选手需要判断论文中的实体之间的语义关系（如致病、治疗、副作⽤等）。
- ⽐赛的成果将被收集整理成公开发布的新冠肺炎研究知识图谱，以及相关技术报告，并可能在药物筛选，辅助医疗，智能搜索，医学知识科普等领域发挥作⽤。

## 数据说明

- **本赛道发布的数据集为COVID-19相关的公开论文文本（英文）**

  - **形式化定义**
    文本样例：COVID-19 is caused by the Severe Acute Respiratory Syndrome Coronavirus-2 (SAR-CoV-2), resulting in symptoms, such as fever, cough, and shortness of breath.

    关系名称填充：
    [ Virus, ( causes ) , Disease ]
    [ Disease, ( induce ) , Phenotype ]

- **数据样例**

```json
{
"text": "Prevalence and clinical features of 2019 novel coronavirus disease (COVID-19) in the Fever Clinic of a teaching hospital in Beijing: a single-center, retrospective study\tBackground With the spread of COVID-19 from Wuhan, Hubei Province to other areas of the country, medical staff in Fever Clinics faced the challenge of identifying suspected cases among patients with respiratory infections manifested with fever. We aimed to describe the prevalence and clinical features of COVID-19 as compared to pneumonias of other etiologies in a Fever Clinic in Beijing. Methods In this single-center, retrospective study, 342 cases of pneumonia were diagnosed in Fever Clinic in Peking University Third Hospital between January 21 to February 15, 2020. From these patients, 88 were reviewed by panel discussion as possible or probable cases of COVID-19, and received 2019-nCoV detection by RT-PCR. COVID-19 was confirmed by positive 2019-nCoV in 19 cases, and by epidemiological, clinical and CT features in 2 cases (the COVID-19 Group, n=21), while the remaining 67 cases served as the non-COVID-19 group. Demographic and epidemiological data, symptoms, laboratory and lung CT findings were collected, and compared between the two groups. Findings The prevalence of COVID-19 in all pneumonia patients during the study period was 6.14% (21/342). Compared with the non-COVID-19 group, more patients with COVID-19 had an identified epidemiological history (90.5% versus 32.8%, P<0.001). The COVID-19 group had lower WBC [5.19×10^9/L (±1.47) versus 7.21×10^9/L (±2.94), P<0.001] and neutrophil counts [3.39×10^9/L (±1.48) versus 5.38×10^9/L (±2.85), P<0.001] in peripheral blood. However, the percentage and count of lymphocytes were not different. On lung CT scans, involvement of 4 or more lobes was more common in the COVID-19 group (45% versus 16.4%, P=0.008). Interpretation In the period of COVID-19 epidemic outside Hubei Province, the prevalence of COVID-19 in patients with pneumonia visiting to our Fever Clinic in Beijing was 6.14%. Epidemiological evidence was important for prompt case finding, and lower blood WBC and neutrophil counts may be useful for differentiation from pneumonia of other etiologies.",
    "spolist": [
    {
        "head": {
        "word": "COVID-19",
        "start": 200,
        "end": 208
        },
        "rel": "induce",
        "tail": {
        "word": "fever",
        "start": 408,
        "end": 413
        }
    },
    {
        "head": {
        "word": "respiratory infections",
        "start": 369,
        "end": 391
        },
        "rel": "induce",
        "tail": {
        "word": "fever",
        "start": 408,
        "end": 413
        }
    },
    {
        "head": {
        "word": "COVID-19",
        "start": 476,
        "end": 484
        },
        "rel": "induce",
        "tail": {
        "word": "Fever Clinic",
        "start": 536,
        "end": 548
        }
    },
...,
]
}
```

## 评测方法

由于每段文本包含多个知识三元组，评测指标使用<实体,关系,实体>而非关系来计算来计算ACC值。

## 时间轴及奖励

**大赛时间轴**

- **2020年06月08日 08:00 ( UTC:00:00 )**
  赛事启动，开放报名、提交。
- **2020年10月31日 07:59 ( UTC:23:59 )**
  初赛提交截止，关闭报名与组队。
- **2020年11月01日 08:00 ( UTC:00:00 )**
  开启复赛，开放测试集下载。
- **2020年11月02日 08:00 ( UTC:00:00 )**
  复赛提交截止。
- **2020年11月02日-11月15日**
  确认最终比赛排名。

**大赛奖励**

**总奖金10万元，本赛道奖金50,000元**

- **一等奖 ￥30,000** 1队
- **二等奖 ￥15,000** 1队
- **三等奖 ￥5,000** 1队