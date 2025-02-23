# Sentiment-Analysis
用Python编写的微调BERT模型做情感极性分类项目，对IMDB影评数据进行数据分析
<img src="https://github.com/xuanzhenhuang/Sentiment-Analysis/blob/main/image/%E6%8A%93%E5%8F%96%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%86%E6%9E%90/1.%E8%AF%84%E5%88%86%E5%88%86%E5%B8%83%E7%9B%B4%E6%96%B9%E5%9B%BE.png?raw=true" width="500px">


## 💡 项目介绍
### 项目背景

随着互联网和社交媒体的快速发展，用户生成内容（UGC）的数量呈爆炸式增长，尤其是在影评、商品评论等领域。这些文本数据中蕴含着丰富的情感信息，能够为企业、研究机构和个人提供有价值的洞察。例如，电影制片方可以通过分析观众对电影的评论来了解观众的情感倾向，从而优化未来的电影制作和营销策略。然而，手动分析海量文本数据不仅耗时耗力，还容易受到主观因素的影响。因此，自动化情感分析技术应运而生，成为自然语言处理（NLP）领域的重要研究方向之一。

情感极性分类是情感分析的一个核心任务，旨在将文本划分为正面或负面等情感类别。传统的机器学习方法在处理复杂语义和上下文依赖时表现有限近年来，预训练语言模型（如BERT）的出现极大地提升了情感分类任务的性能。BERT（Bidirectional Encoder Representations from Transformers）通过双向Transformer架构捕捉文本的上下文信息，能够更好地理解文本的语义和情感倾向。

本项目旨在利用BERT模型对IMDB影评数据集进行情感极性分类。IMDB数据集是情感分析领域的经典数据集，包含50,000条电影评论，每条评论被标记为正面或负面。通过对这些评论进行数据分析，我们可以深入了解用户对电影的情感倾向，并进一步微调BERT模型以实现高精度的情感分类。此外，为了验证模型的泛化能力，我们还从互联网上抓取了2021年至2025年的10部电影累计921条最新影评数据，用于模型的二次验证。

---

### 项目流程

1. **数据预处理**  
   对IMDB影评数据进行清洗和格式化处理，以适应 BERT 模型的输入要求。通过各种数据处理库（如 Pandas、Re、Nltk）将所有的txt影评文件处理在一个excel表内、去除HTML标签、特殊字符、停用词等，并将数据顺序打乱。

2. **数据预览**  
   对IMDB数据集进行初步探索性分析（EDA），包括词频统计、评论长度统计、情感标签分布等。通过可视化工具（如Matplotlib、Seaborn）展示数据的基本特征。

3. **模型微调**  
   基于预训练的BERT模型，使用 IMDB 数据集进行微调，以适应情感分类任务。具体步骤包括：
   - 加载预训练的BERT模型（`bert-base-uncased`），初始化分类模型（`BertForSequenceClassification`），适用于二分类任务。
   - 使用 `AdamW` 优化器，适用于 BERT 模型的微调。
   - 调用GPU加速模型的训练和验证，对模型性能进行评估并计算准确率。

4. **数据抓取**  
   从IMDB官网上抓取2021年至2025年的921条最新影评数据，用于验证模型的泛化能力。抓取的数据包括电影编号、评论内容、评分、评论人数、发布时间等信息。

5. **数据处理**  
   对抓取的最新影评数据进行清洗和格式化处理。包括去除噪声数据、统一文本格式、进行情感标签标注。

6. **数据分析**  
   对抓取的最新影评数据进行探索性分析，比较其与IMDB数据集的差异。分析内容包括评分分布情况、点赞数和差评数的相关性、不同时间段的评价数量变化趋势、词频统计以及词云图可视化分析等。

7. **模型预测**  
   使用微调后的BERT模型对抓取的最新影评数据进行情感分类预测，并评估模型在新数据上的性能。

8. **总结**  
   总结项目的整体流程和结果，分析模型在IMDB数据集和最新影评数据上的表现差异。讨论模型的优缺点，并提出未来改进方向（如引入更多数据增强技术、尝试其他预训练模型等）。

---

### 项目亮点

1. **经典与最新数据的结合**  
   不仅使用了经典的IMDB数据集，还引入了最新的影评数据，验证模型在真实场景中的泛化能力。

2. **完整的NLP流程**  
   从数据预处理、模型微调、数据抓取到模型预测，涵盖了自然语言处理项目的全流程。

3. **技术深度**  
   使用BERT模型进行微调，展示了深度学习在情感分析任务中的强大能力。

4. **实际应用价值**  
   通过情感分析，为电影制片方、营销团队等提供了有价值的用户洞察。






### 示例代码片段

#### 数据预处理
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    # 清洗文本
    text = text.lower().strip()
    # Tokenization
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    return tokens
```

#### 数据抓取
```python
import requests
from bs4 import BeautifulSoup

def scrape_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    reviews = soup.find_all('div', class_='review-text')
    return [review.get_text() for review in reviews]
```

#### 模型微调
```python
from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练循环
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

---

通过以上流程和代码示例，项目不仅展示了BERT模型在情感分类任务中的应用，还验证了其在新数据上的泛化能力，具有较强的实用性和技术深度。
