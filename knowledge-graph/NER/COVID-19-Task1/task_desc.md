# COVID-19 医学论文实体抽取

## 简介



本次⽐赛聚焦于医疗（特别是和新冠肺炎相关）的知识图谱技术。数据来源包括近期发布的有关新冠肺炎的英文学术论⽂数据，并经过标注。

- 比赛共分为两个赛道。
  [赛道一任务为：医学论文实体识别](https://www.biendata.com/competition/chaindream_knowledgegraph_19_task1/)，参赛选⼿需要从中抽取出指定类型的实体（例如疾病、症状、病毒、基因、药物等）。
  [赛道二任务为：医学论文关系抽取](https://www.biendata.com/competition/chaindream_knowledgegraph_19_task2/)，参赛选手需要判断论文中的实体之间的语义关系（如致病、治疗、副作⽤等）。
- ⽐赛的成果将被收集整理成公开发布的新冠肺炎研究知识图谱，以及相关技术报告，并可能在药物筛选，辅助医疗，智能搜索，医学知识科普等领域发挥作⽤。

## 数据说明

**本赛道发布的数据集为COVID-19相关的公开论文文本（英文）**

- **输入**
  - 论文的题目与摘要文本集合：
    D={d1 ,…,dN}, di = {wi = i1,…,win}
  - 预定义类别：C={c1 ,…,cm}
- **输出**
  实体和所属类别对的集合：{< m1 , c1 > , < m2, c2 >, …, < mp,cm >}
  其中mi=< di , bi , ei >是出现在文本中的医学实体di，bi 和ei分别表示在中的起止位置，cm∈C表示所属的预定义类别。
- **预定义类别**（举例）
  - 疾病：医学上定义的疾病，如COVID-19，心肌炎等。
  - 症状：疾病中表现出的临床症状，如发热、咳嗽，胸闷等。
  - 药物：用于疾病治疗的具体化合药物或中药，如：瑞德西韦，连花清瘟胶囊等。
  - 化合物：用以合成药物的化学元素。
  - 基因：核苷酸序列，如：ACE2等
  - 病毒：一些引发疾病的微生物，如：SARS-Cov-2、SARS等。

- **数据样例**

```json
{
"text": "Papillomavirus vaccines in clinical trials\tCervical cancer remains a leading cause of death for women in the developing world, and the treatment of preneoplastic cervical lesions is a considerable public-health burden in the developed world. There is unambiguous evidence that human papillomaviruses (HPVs) trigger the development of cervical and other anogenital malignancies, and that continued expression of HPV antigens in the tumours drives the neoplastic progression. The viral cause of cervical cancer is also its Achilles heel. Prophylactic vaccines to prevent HPV infection and therapeutic vaccines targeted at the HPV tumour antigens are in clinical trials.",
    "entities": [
        {
        "entity": "Papillomavirus vaccines",
        "type": "Drug",
        "start": 0,
        "end": 23
        },
        {
        "entity": "Cervical cancer",
        "type": "Disease",
        "start": 43,
        "end": 58
        },
        {
        "entity": "HPVs",
        "type": "Virus",
        "start": 301,
        "end": 305
        },
        {
        "entity": "anogenital malignancies",
        "type": "Disease",
        "start": 353,
        "end": 376
        },
        {
        "entity": "HPV antigens",
        "type": "Gene",
        "start": 411,
        "end": 423
        },
        {
        "entity": "tumours",
        "type": "Disease",
        "start": 431,
        "end": 438
        },
        {
        "entity": "cervical cancer",
        "type": "Disease",
        "start": 493,
        "end": 508
        }
    ]
}
```

## 评测方法

本任务采用精确率（Precision）、召回率（Recall）以及**F1-Measure**作为评测指标。
TP:正例预测正确的个数；FP：负例预测错误的个数；TN：负例预测正确的个数；FN：正例预测错误的个数
![img](https://www.biendata.xyz/media/competition/2020/06/08/15915915676639512%E6%9C%AA%E6%A0%87%E9%A2%98-1.jpg)

具体来说，对预定义的8个不同类别，对每个子类进行分开评测，最后以平均值进行排名。

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