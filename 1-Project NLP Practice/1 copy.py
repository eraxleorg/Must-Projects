import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


nltk.data.path.append("stopwords")
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove special characters, numbers, and punctuation
    text = re.sub(r"[^a-z\s]", "", text)

    # 3. Tokenize the text
    words = word_tokenize(text)

    # 4. Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Return the cleaned text as a single string
    return " ".join(words)


resumes = [
    {
        "resume": "Experienced software engineer with 5 years of experience in Python, Django, REST APIs, and database management. Worked on developing scalable web applications and improving backend performance.",
        "job_role": "Software Engineer",
    },
    {
        "resume": "Certified public accountant with expertise in financial reporting, auditing, tax preparation, and regulatory compliance. Proficient in using accounting software such as QuickBooks and Excel.",
        "job_role": "Accountant",
    },
    {
        "resume": "Project manager with 10 years of experience in leading teams, managing budgets, and ensuring timely project delivery. Expertise in agile methodologies, risk management, and client communication.",
        "job_role": "Project Manager",
    },
    {
        "resume": "Digital marketing specialist with a focus on social media strategy, SEO, content marketing, and paid advertising. Managed multiple campaigns that improved brand visibility and customer engagement.",
        "job_role": "Digital Marketing Specialist",
    },
    {
        "resume": "Mechanical engineer with hands-on experience in designing mechanical systems, CAD modeling, and performing stress analysis. Strong understanding of thermodynamics and manufacturing processes.",
        "job_role": "Mechanical Engineer",
    },
    {
        "resume": "Human resources specialist with a background in employee recruitment, performance management, and benefits administration. Implemented onboarding programs that increased employee retention rates.",
        "job_role": "HR Specialist",
    },
    {
        "resume": "Data scientist with a focus on machine learning, data analysis, and visualization. Experience with Python, R, and SQL to derive insights from large datasets and develop predictive models.",
        "job_role": "Data Scientist",
    },
    {
        "resume": "Customer service representative with excellent communication skills and a track record of resolving customer issues efficiently. Experience in handling customer inquiries, complaints, and support requests.",
        "job_role": "Customer Service Representative",
    },
    {
        "resume": "Graphic designer skilled in Adobe Photoshop, Illustrator, and InDesign. Created visual designs for digital and print media, including branding, advertisements, and web content.",
        "job_role": "Graphic Designer",
    },
    {
        "resume": "Sales executive with a proven track record in B2B sales, lead generation, and closing deals. Adept at developing sales strategies, building client relationships, and achieving targets.",
        "job_role": "Sales Executive",
    },
]

preprocessed_resumes = [preprocess_text(resume["resume"]) for resume in resumes]
job_roles = [job_role["job_role"] for job_role in resumes]
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(preprocessed_resumes)

le = LabelEncoder()
print(dir(le))


y = le.fit_transform(job_roles)

# Step 5: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Step 6: Train a model (Naive Bayes as an example)
model = MultinomialNB()
model.fit(X_train, y_train)


# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(classification_report(y_test, y_pred))







# Real-time resume data
new_resume = """- Create visually stunning designs for digital and print materials, including websites, social media graphics, brochures, advertisements, and more

- Collaborate with the marketing team to develop and execute creative concepts that align with our brand guidelines and objectives

- Understand project requirements and deliver designs that effectively communicate the desired message

- Use industry-standard design software and tools to create high-quality graphics and layouts

- Stay up to date with design trends and best practices to ensure our designs are modern and relevant

- Work closely with the team to ensure projects are delivered on time and meet the required quality standards

- Incorporate feedback and make revisions to designs based on stakeholder input

- Maintain organized files and documentation of design assets for easy access and future reference"""

# Step 1: Preprocess the new resume using the same preprocessing function
processed_resume = preprocess_text(new_resume)

# Step 2: Vectorize the new resume using the same vectorizer (TF-IDF)
new_resume_vector = vectorizer.transform([processed_resume])

# Step 3: Predict the job role using the trained model
predicted_label = model.predict(new_resume_vector)

# Step 4: Convert the predicted numeric label back to the job role using the label encoder
predicted_job_role = le.inverse_transform(predicted_label)

print("Predicted job role:", predicted_job_role[0])






# import pandas as pd

# # Convert the sparse matrix to a dense array
# dense_array = x.toarray()

# # Get the feature names (if using CountVectorizer or TfidfVectorizer)
# feature_names = vectorizer.get_feature_names_out()

# # Create a DataFrame from the dense array
# df = pd.DataFrame(dense_array, columns=feature_names)

# # Export the DataFrame to a CSV file
# df.to_csv('resumes_vectorized.csv', index=False)





