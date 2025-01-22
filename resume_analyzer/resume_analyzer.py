import streamlit as st
import requests
import json
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from datetime import datetime
import google.generativeai as genai
import os
from dateutil import parser

def extract_text_from_pdf(pdf_file):
    """Extracts text from uploaded PDF (optimized for text-based PDFs)."""
    try:
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def extract_information_llm(text, instructions, model_name='gemini-pro'):
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(model_name)

        prompt = f"### Instruction:\n{instructions}\n\n### Input:\n{text}\n\n### Response:"
        response = model.generate_content(prompt)
        if "json" in response.text or "JSON" in response.text:
            cleaned_json_output = response.text.replace("```", "")
            cleaned_json_output = cleaned_json_output.replace("JSON", "")
            cleaned_json_output = cleaned_json_output.replace("json","")
            #print(cleaned_json_output)
            return cleaned_json_output
        else:
            return response.text
    except Exception as e:
        st.error(f"Error processing LLM response: {e}.")
        return None


# Define the parse_date functioni
def parse_date(date_str):
    if date_str == "Present":
        return datetime.now()
    else: # Define the format to match the date string 
        # date_format = "%B %Y"
        # return datetime.strptime(date_str, date_format)
	return parser.parse(date_str)

def draw_charts(experiences, skills_data):
    try:
        # Convert dates to datetime objects and calculate duration in years
        dates = [(entry['Job Title'], entry['Company'], parse_date(entry['Start Date']), parse_date(entry['End Date'])) for entry in experiences]
        durations = [(job, company, (end - start).days / 365) for job, company, start, end in dates]

        # Job titles with company names for labels
        labels = [f"{job}\n{company}" for job, company, _ in durations]

        # Extracting durations for plotting
        values = [duration for _, _, duration in durations]

        # Predefined set of 30 distinguishable colors
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#ffcc7f', '#8dd3c7', '#fb8072', '#b3de69', '#fdb462',
            '#80b1d3', '#b15928', '#d9d9d9', '#bc80bd', '#ccebc5',
            '#ffed6f', '#c7eae5', '#f4cae4', '#f1e2cc', '#cccccc'
        ]


        # Plotting the bar chart
        st.subheader('Work Experience Timeline')
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors[:len(values)], edgecolor='k')

        # Adding text annotations for duration in years
        for bar, duration in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height - 0.5, f'{duration:.1f} years\n(-)', ha='center', va='top', fontsize=10, fontweight='bold')

        # Adding the years on the x-axis
        start_year = 2014
        end_year = datetime.now().year + 1  # Adding one year to include the current year
        years = list(range(start_year, end_year))

        # Set the x-axis ticks to match the number of bars
        ax.set_xticks([i for i in range(len(labels))])
        ax.set_xticklabels(labels, rotation=45, ha='right')

        # Adding the years as additional x-axis labels
        #ax2 = ax.twiny()  # Create a secondary x-axis
        #ax2.set_xlim(ax.get_xlim())
        #ax2.set_xticks(range(len(years)))  # Match the number of ticks to the number of years
        #ax2.set_xticklabels([str(year) for year in years], rotation=0, ha='center')

        # Labels and title
        ax.set_xlabel('Job Titles and Companies')
        ax.set_ylabel('Duration in Years')
        ax.set_title('Work Experience Bar Chart')

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Preparing pie chart data
        skill_labels = list(skills_data.keys())
        skill_sizes = list(skills_data.values())

        # Plotting the pie chart
        st.subheader('Skill Distribution')
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(skill_sizes, labels=skill_labels, autopct='%1.1f%%', colors=colors[:len(skill_labels)], startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Title
        ax.set_title('Skill Distribution Based on Work Experience')

        # Display the pie chart in Streamlit
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error experience/skills chart: {e}.")
        st.write(experiences)
        st.write(skills_data)
        return None



def calculate_match_llm(extracted_resume_info, job_description_json, model_name):
    """Calculates match using Ollama API and JSON job description."""
    sample_result_json = """
        {
          "score": <score>,
          "assessment": <assessment>,
          "reason": <reason>,
          "questions": [
            {"question": <Question 1>},
            {"question": <Question 2>}
           ]
        }
    """
    instructions = f"""
    Given the following resume information:\n{extracted_resume_info}\nAnd this job description in JSON format:\n{json.dumps(job_description_json)}
    \nProvide a match score out of 5 and a suitability assessment ('Highly Suitable', 'Moderately Suitable', or 'Not Suitable'), along with reason the score. 
    Generate a JSON array of 5 conceptual questions and 5 coding questions based on the candidate's skills which should not contain any other details. 
    Make sure there is consistency between score and suitability assessment for example more suitable candidate will have higher score. 
    Respond in strict parsable JSON format like below which will be used directly in a JSON parser :

    {sample_result_json}
    """

    match_result = extract_information_llm("", instructions, model_name)
    #st.write(instructions)
    #st.write(match_result)
    try:
        match_json = json.loads(match_result)
        return match_json
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Could not parse LLM match responsei : {e}. Returning default values.")
        st.write(match_result)
        return {"score": 0, "assessment": "Not Suitable"}

st.title("Resume Analyzer & Recommender")

model_name = 'gemini-pro' #st.selectbox("Select Model", ["llama3.2:3b-instruct-q4_K_M", "mistral:7b-instruct","llama3.2:1b", "llama2:7b-chat", "llama2:13b", "llama2:7b"])

job_description_json_str = st.text_area("Enter/modify job requirements in JSON format.","""
{
        "title": "Network/System Engineer",
        "description": "As a Network/System Engineer, you will work on developing or fixing issues complex and large Networking Operating software, debugging customer issues. Need to work with multiple people or teams and write design document, code, unit testing and also participate in review etc.",
        "skillsRequired": "C, data structure, Networking basics, linux/system programming, debugging, problem solving, problem analytics, proactive learner, positive attitude",
	"location": "Hyderabad",
        "employmentType": "Full-time"
}
""", height=300)
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])


try:
    if job_description_json_str:
        job_description_json = json.loads(job_description_json_str)
    else:
        job_description_json = None
        st.warning("Please enter a valid JSON job description.")
except json.JSONDecodeError as e:
    st.error(f"Invalid JSON format for job description. {e}")
    st.write(job_description_json_str)
    job_description_json = None

if uploaded_file and job_description_json:
    resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text:
        sample_resume_json ="""
            {
              "Name": "John Doe",
              "Contact": {
                "Address": "Finland...",
                "Phone": "+358-50-9876543",
                "Email": "john.doe@example.com"
              },
              "Summary/Objective": "A versatile researcher ...",
              "Experience": [
                {
                  "Job Title": "Senior Researcher",
                  "Company": "Helsinki Institute of Technology, Finland",
                  "Start Date": "April 2018",
                  "End Date": "Present"
                }
               ],
              "Education": [
                {
                  "Degree": "Ph.D. in Computer Science",
                  "University": "University of Turku, Finland",
                  "Graduation Date": "December 2013"
                }
              ],
              "Skills": [
                  {
                    "Skill": "Designing, managing and executing research projects",
                    "Weightage": 80
                  },
                  {
                    "Skill": "Testing",
                    "Weightage": 20
                  }
               ]
            }
        """
        instructions_resume = f"""
        Extract the following information from the resume: Name, Contact Information, Summary/Objective, Experience (Job Title, Company, Dates), Education (Degrees, Universities), Skills.
        The "Skills" should be having skills details and its weightage percentage based on usage in different projects and how many or how long it has been used, the sum of all weightages should be 100%.
        Make sure to create JSON having following format such that it can be directly fed directly to a JSON parser:

        {sample_resume_json}    
        """
        extracted_resume_info = extract_information_llm(resume_text, instructions_resume, model_name)

        if extracted_resume_info:
            st.write("\nExtracted Resume Information:")
            try:
                resume_json = json.loads(extracted_resume_info)

                # Draw chart 
                skills_arr = resume_json["Skills"]

                skills_dict = {}
                for skill in skills_arr:
                    if (skill["Skill"] is None):
                        skills_dict[skill["Skill"]] = 100/len(skills_arr)
                    else:
                        skills_dict[skill["Skill"]] = skill["Weightage"]

                draw_charts(resume_json["Experience"], skills_dict)

            except json.JSONDecodeError as e:
                st.write(f"Could not parse extracted resume info as JSON {e}. Raw output:")
                st.write(extracted_resume_info)

            match_data = calculate_match_llm(extracted_resume_info, job_description_json, model_name)
            st.markdown("## Match Result :")
            for key, value in match_data.items():
                if isinstance(value, list):
                    st.markdown(f"**{key.capitalize()}:**")
                    for i, item in enumerate(value):
                        st.markdown(f"**Question {i+1}:** {item['question']}")
                else:
                    if "score" in key:
                        st.markdown(f"**{key.capitalize()}:** {value} out of 5")        
                    else:
                        st.markdown(f"**{key.capitalize()}:** {value}")        

        else:
            st.error("Could not extract information from the resume using the LLM.")

    else:
        st.error("Could not process the resume PDF.")

st.write("\nDisclaimer: This is an LLM-based matching system. Results may vary and human review is always recommended.")
