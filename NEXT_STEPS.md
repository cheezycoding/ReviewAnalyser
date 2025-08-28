# Next Steps: Restaurant Review Policy Detection with Autolabel
## TikTok TechJam Hackathon - ML Pipeline Continuation

### 🎯 **Current Status: READY FOR AUTOLABEL SETUP**

**✅ COMPLETED:**
- **1,140 restaurant reviews** collected from 15 Singapore restaurants
- **12 hand-curated seed examples** labeled across 4 categories
- **Comprehensive labeling schema** and guidelines created
- **Data analysis** and quality assessment completed

---

## 📊 **Seed Dataset Summary**

### **Location:** `data-collection/data/seed_examples.json`
**12 examples across 4 categories:**

#### **AUTHENTIC (3/3) - Genuine personal reviews**
1. **joe fong** (Burnt Ends) - Personal narrative, casual tone, "hahaha"
2. **Amy Hoon** (My Awesome Cafe) - Enthusiastic, personal interactions, emoticons  
3. **Richard De La Rosa** (Burnt Ends) - Personal bar story, specific interactions

#### **FAKE (3/3) - AI-generated/marketing content**
1. **Krison Tan** (Burnt Ends) - Overly polished, marketing-style language
2. **Danielle Perrow** (Burnt Ends) - "Delights the taste buds", promotional language
3. **Shim Kl** (My Awesome Cafe) - Generic templates, "culinary masterpiece", no specifics

#### **LOW_QUALITY (3/3) - Uninformative content**
1. **Aparna** - "Best indian food in boat quay" (29 chars)
2. **Krzysztof** - "Great pork prawn noodle" (23 chars)
3. **O O** - "Good food, hot place." (19 chars)

#### **IRRELEVANT (3/3) - Off-topic content**
1. **Delivery logistics** (synthetic) - About delivery problems, not restaurant
2. **Parking complaints** (synthetic) - Infrastructure issues + personal rambling
3. **Neighborhood nostalgia** (synthetic) - Area history, unrelated to dining

---

## 🚀 **IMMEDIATE NEXT STEPS**

### **Step 1: Install and Setup Autolabel**
```bash
# Install autolabel
pip install refuel-autolabel

# Set up OpenAI API key (or other LLM provider)
export OPENAI_API_KEY=your_api_key_here

# Verify installation
python -c "from autolabel import LabelingAgent; print('✅ Autolabel ready!')"
```

### **Step 2: Create Autolabel Configuration**
Create `autolabel_config.json` in the project root:

```json
{
  "task_name": "RestaurantReviewPolicyDetection",
  "task_type": "classification",
  "model": {
    "provider": "openai",
    "name": "gpt-4"
  },
  "dataset": {
    "label_column": "label",
    "delimiter": ","
  },
  "prompt": {
    "task_guidelines": "You are an expert at analyzing restaurant reviews for policy violations and quality assessment. Classify each review into one of the following categories: {labels}",
    "labels": [
      "authentic",
      "fake", 
      "low_quality",
      "irrelevant"
    ],
    "few_shot_examples": [
      {
        "example": "Finally after months of waiting, secured a meal in this popular restaurant..\n\nBurnt ends, actually served beyond steak as their signature..but smoked flavours in all their cuisine.\n\nGrissini, crisp breadsticks, paired with taramasalata, a traditional Greek dip made from salted cod roe, olive oil, lemon juice, and crispy flat bread was refreshing great appetising starter.\n\nQuail egg smoked giving an explosive smoky flavour on one bite, caviar was a great compliment to the texture and brings freshness to the smokeness.\n\nJamaica wings or rather a halved wing, looks too small for me, but it took me 3 bites , hahaha. Is because I totally loved that flavour and slowly tasted the wonderful wings flavour.\n\nEel and bone marrow, like the photos shows..seems normal like a unagi sushi , but this is done smoked flavour glazed in the special miso sauce that I consider 5 🌟 against any omakase japanese resturant.   The bone marrow was hidden as a base that did not really over power the eel.\n\nMain course was WA Marron grill to perfection, like medium rare..that is preserved the softness of the body, the glazed sauce was amazing don't even remember what was it.\n\nFinally the Australian steak, it was ok, lack the punch of a wow for a steak\n\nConcluded with smoked lava cake, this is totally amazingly different with a strong smoky smoke in ur mouth,  of course it has to be eaten with the ice cream.\n\nI am thrilled by the different presentation of course that resembles smoky identity yet letting each element of food be amazingly created to refine texture and taste.",
        "label": "authentic"
      },
      {
        "example": "Stepping into this restaurant immediately sets the tone for a refined yet warmly inviting dining experience. The high ceilings, exposed beams, and striking vertical pillars give the space a grand industrial-chic feel, complemented by warm amber lighting and neatly set tables. The open kitchen allows diners to watch the chefs in action, adding an engaging, lively atmosphere.\n\nThe food presentation is elegant and minimalist, allowing the quality of the ingredients to shine. The slices of beef are cooked to a perfect medium-rare, with a delicate char on the outside and a juicy, tender center, paired simply with fresh greens for a clean, balanced plate. Another standout is the richly glazed beef-topped toast, crowned with tart pickles and fresh chives—a harmonious combination of sweet, tangy, and savory notes.\n\nThe playful touch of charred marshmallows on skewers hints at the restaurant's creative flair, blending nostalgic comfort with gourmet technique.\n\nService is very attentive and professional, with the chefs clearly passionate about their craft. Overall, this is a place that blends sophisticated dining with a relaxed ambience, making it perfect for both special occasions and indulgent casual meals.",
        "label": "fake"
      },
      {
        "example": "Best indian food in boat quay",
        "label": "low_quality"
      },
      {
        "example": "The delivery guy was 30 minutes late and I had to call the delivery company twice. My apartment is hard to find and the GPS doesn't work properly in this area. The driver didn't speak English well. Very frustrating experience with the delivery service.",
        "label": "irrelevant"
      }
    ],
    "example_template": "Input: {example}\nOutput: {label}"
  }
}
```

### **Step 3: Prepare Training Dataset**
Convert reviews to CSV format for autolabel:

```python
import json
import pandas as pd

# Load reviews
with open('data-collection/data/reviews_clean.json', 'r') as f:
    reviews = json.load(f)

# Create training dataset
training_data = []
for review in reviews:
    training_data.append({
        'review_id': review['review_id'],
        'restaurant_name': review['restaurant_name'], 
        'review_text': review['review_text'],
        'review_rating': review['review_rating'],
        'author': review['author'],
        'label': ''  # Empty for autolabel to fill
    })

# Save as CSV
df = pd.DataFrame(training_data)
df.to_csv('training_dataset.csv', index=False)
print(f"✅ Created training dataset with {len(df)} reviews")
```

### **Step 4: Run Autolabel**
```python
from autolabel import LabelingAgent, AutolabelDataset

# Initialize agent
agent = LabelingAgent(config='autolabel_config.json')

# Load dataset
ds = AutolabelDataset('training_dataset.csv', config=agent.config)

# Preview and estimate costs
agent.plan(ds)

# Run labeling (this will cost API credits)
ds = agent.run(ds)

# Save results
ds.df.to_csv('labeled_reviews.csv', index=False)
print(f"✅ Labeled {len(ds.df)} reviews!")
```

---

## 📋 **SUBSEQUENT STEPS**

### **Step 5: Model Development**
1. **Feature Engineering** - Extract text features, metadata features
2. **Train ML Models** - Use labeled data to train classifiers
3. **Evaluation** - Test on held-out data, calculate metrics

### **Step 6: Policy Enforcement Module**
1. **Threshold Tuning** - Set confidence thresholds for each category
2. **Rule-based Filtering** - Implement business logic for policy violations
3. **API Development** - Create inference endpoint

### **Step 7: Evaluation & Reporting**
1. **Metrics Calculation** - Precision, recall, F1-score by category
2. **Error Analysis** - Review misclassified examples
3. **Performance Report** - Document findings and recommendations

---

## 🎯 **Key Files & Locations**

- **Seed Examples**: `data-collection/data/seed_examples.json`
- **Raw Reviews**: `data-collection/data/reviews_clean.json` (1,140 reviews)
- **Autolabel Config**: `autolabel_config.json` (to be created)
- **Training Dataset**: `training_dataset.csv` (to be created)
- **Labeled Results**: `labeled_reviews.csv` (to be created)

---

## ⚠️ **Important Notes**

1. **API Costs**: Autolabel will use OpenAI API credits (~$5-20 for 1,140 reviews)
2. **Quality Assurance**: Review autolabel results, especially low-confidence predictions
3. **Model Selection**: Consider using GPT-4 for best accuracy, GPT-3.5 for cost efficiency
4. **Validation**: Spot-check ~50 autolabeled reviews to ensure quality

---

## 🚀 **Success Metrics**

- **Labeling Accuracy**: >85% agreement with human labels
- **Coverage**: All 1,140 reviews labeled
- **Cost Efficiency**: <$20 total API costs
- **Time**: <2 hours for complete labeling

---

**Status**: ✅ Ready for autolabel execution
**Next Agent**: Proceed with Step 1 (Install & Setup)
