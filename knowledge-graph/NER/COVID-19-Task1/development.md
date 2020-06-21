# COVID-19  NER开发文档

### TODO

- [x] ~~比赛所提供数据分析，包括标注情况、实体类别、总数据量~~

- [x] ~~确定数据处理内容，包括分词、停用词、stem等操作~~

- [ ] 对数据集进行划分为train/dev/test

- [ ] 算法、框架选择和实现

- [ ] 效果评估、代码优化

  



## 1 任务分析

Task1 主要任务是从给定的文本中提取出COVID-19相关的实体，文本为论文标题和摘要集合。实体类型有，

- 疾病（Disease）：医学上定义的疾病，如COVID-19，心肌炎等。
- 症状（Phenotype）：疾病中表现出的临床症状，如发热、咳嗽，胸闷等。
- 药物(Drug)：用于疾病治疗的具体化合药物或中药，如：瑞德西韦，连花清瘟胶囊等。
- 化合物(ChemicalCompound)：用以合成药物的化学元素。
- 化学品(Chemical)：
- 基因(Gene)：核苷酸序列，如：ACE2等
- 病毒(Virus)：一些引发疾病的微生物，如：SARS-Cov-2、SARS等
- Organization: 组织机构

**任务特点**

1. 数据比较新颖，与COVID-19相关论文paper文本中抽取相关实体，同一个实体表述形式存在多样
2. 实体词结构形式多样，有复数形式、连词形式、长词形式
3. 数据标注存在一定错误性

## 2 数据分析

经过对task1_train.json分析

**数据格式**

数据以json形式提供，例子如下，

```json
{"text": "Health security in 2014: building on preparedness knowledge for emerging health threats\tIdeas, information, and microbes are shared worldwide more easily than ever before. New infections, such as the novel influenza A H7N9 or Middle East respiratory syndrome coronavirus, pay little heed to political boundaries as they spread; nature pays little heed to destruction wrought by increasingly frequent natural disasters. Hospital-acquired infections are hard to prevent and contain, because the bacteria are developing resistance to the therapeutic advances of the 20th century. Indeed, threats come in ever-complicated combinations: a combined earthquake, tsunami, and radiation disaster; blackouts in skyscrapers that require new thinking about evacuations and medically fragile populations; or bombings that require as much psychological profiling as chemical profiling.", "entities": [{"entity": "influenza A", "type": "Virus", "start": 206, "end": 217}, {"entity": "H7N9", "type": "Disease", "start": 218, "end": 222}, {"entity": "Middle East respiratory syndrome coronavirus", "type": "Virus", "start": 226, "end": 270}]}
```

整个json串有两部分内容构成，"text"和"entites",前者是文本有标题和摘要构成，二者之间用\t来分割，"entities"字段表示该文本中所出现的所有实体列表，包括实体，实体类型，实体在文本中出现的位置。

所给实体，对于同一实体词，即有大写字母开头形式，也有小写字母开头形式，单复数形式均有，长度不限。

**实体类型分布**

```json
{'Drug': 11017, 'Disease': 28300, 'Gene': 12542, 'ChemicalCompound': 65, 'Virus': 6827, 'Chemical': 2057, 'Phenotype': 3066, 'Organization': 1719}
```

其中key为实体类型，value为含有该类型的文档总数。很明显实体类型分布是严重不均衡的，因此后续在处理时需要考虑不均衡问题。

对文件中出现的实体词进行统计，共有15575个实体词。

**数据总量**

task1_train.json文件总共9679条数据，数据量比较少。如果要切分train/dev文件，则将长文本拆分为多个句子会更加合适，进而划分train/dev数据集。测试文件task1_valid_noAnswer.json包含4839条数据

**总结**

1. 标注数据同一单词存在首字母大、小写两种形式，如”zoonotic“和"Zoonotic","ZIKA"和"Zika","Zika Virus"和”Zika virus“,

2. 实体词存在单复数不同标注，如"younger age"和“younger ages“, "zoonosis"和"zoonoses"（复数形式）

3. 错误标注，如“YouTube”，“Wikipedia”被标注为"Chemical"，数据如下：

   1.样本序号为：数据中出现的YouTube，Wikipedia均可认为是错误标注，<span style='color:red'>错误数据直接删除错误标注</span>

   2.序号为**1495**，出现错误实体标注，"entities":[{"entity":"COVID19","type":"Chemical","start":33,"end":40},{"entity":"SETTINGS","type":"Disease","start":70,"end":78},{"entity":"APP","type":"Gene","start":102,"end":105},{"entity":"EXCEL","type":"Drug","start":110,"end":115},{"entity":"covid19 disease","type":"Disease","start":140,"end":155},{"entity":"covid19 disease","type":"Disease","start":625,"end":640},{"entity":"government measures","type":"Organization","start":798,"end":817}]， 将SETTING标注为disease， EXCEL标注为Drug

   在标注数据中仍存在很多类似标注错误，也许是数据提供方在对数据进行规则提取时产生。

4. 标注数据中，若同一个实体词在文本中出现多次，则全部提取出来

5. 实体词中存在连词符“-”，如hepatitis-C-virus infection中的

6. 所给数据标注数据存在一个情况，即同一个词存在两次标注，比如“<span style='color:red'>refractory epilepsy </span>and normal conventional MRI was examined with diffusion”，标注为[{"entity": "epilepsy", "type": "Disease", "start": 224, "end": 232},{"entity": "refractory epilepsy", "type": "Disease", "start": 213, "end": 232}]，标红词存在二次标注，将其转换为BIO序列标签时，存在错误，因为第二个词是单独的是B-，二在组合词中标签是I-，这种情况的比例在6%，有600多条类似情况。

**数据处理**

在对数据做具体处理时，需要考虑如下问题，

1. 不对英文文本进行大小写转换，保持现有标注形式。(*后续需验证，统一转为某种形式是否对结果有怎样影响*)

2. 删除部分明显标注错误实体

3. 构建统一词典时，将所有数字统一转换为**NUM**字符，主要目的是弱化具体数值带来的信息缺失，凸显数字特性。

4. 按照句子结束符号**.;\t**进行断句，\t是用来划分标题和摘要

5. 切分后的句子，采用BIO进行标注，B-实体词开始，I-实体词中间部分，O-其他，按照这种形式统一构造模型训练数据集

6. 将重新构造后的数据集按照8:2划分为train/dev

7. 对于数据中存在重叠标注情况，删除重叠标注，只保留最长实体所对应的标签

   

   



## 3 算法实现

### 3.1 基于词典匹配

为探索基本模型，方法一是利用task1_train.json中的所有实体词，形成实体词库，采用文本匹配的方式进行实体词抽取，在task1_valid_noAnswer.json中试探所给测试文件中含有实体词的大小，目的是可以确认测试文件中所给实体是多余训练文件中实体词集合。

实现方案是才用AC自动机算法，将文本中的所有实体词提取出来。

**实现过程**

在利用词库匹配过程中发现如下问题，

1. 实体词中有很短的词，如alt,nation等样子，利用字符串匹配算法，如果字符串中出现该组合则即匹配出来，但在原文本中这些词非独立词汇，
2. 字符串匹配时，需要解决最长匹配问题，即只保留最长的那个组合
3. 在文本中提取出的实体词，在原文本语境中是非实体词，如"thinking"标注为Disease,但在文本中是"that require new thinking about ..."

以上三点是利用实体词库进行匹配时，出现的主要问题，换而言之，实体抽取在当下任务下需要通过模型完成，即考虑上下文语境来完成。

**结论**

本计划先通过实体词匹配方式完成一版进行结果提交，但从实体词匹配的结果来看，效果很差，因此基于词典匹配方式可以选择放弃。



### 3.2 基于模型抽取

**思路**

1. 先对清理后数据划分为训练集train、开发机dev
2. 基于BIO规则修改标注格式
3. 构建vocabulary 和label集合
4. 搭建模型Bi-lstm+CRF作为baseline
5. 调研如何将已有实体词库使用起来
6. 后续计划self-attention+CRF->bert finetune+Bi-lstm+CRF

## 4 效果评估

TO BE ADD!



