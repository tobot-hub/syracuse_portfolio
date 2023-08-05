import openai
import pandas as pd

openai.api_key = "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"

class Grader:
    def __init__(self):
        self.description = "Using all the knowledge of a hiring manager and senior data analyst, you judge the probability of a resume being selected for a giving job posting."
        

    def grade_resume(self, resume, job_post):
        prompt = """Based on the resume and job description below, determine the probability that the resume is selected for hire with all other factors being equal.
        Only answer a numeric probability between 0 and 1. Do not include any other non-structured text in your response.
        
        Resume: {}
        Job description: {}
        
        """.format(resume, job_post)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.description},
                {"role": "user", "content": prompt}
            ]  
        )
        result = response.choices[0].message.content
        return result

def main():
    resume = """
        Experience	Change Integrator (Systems Engineer), Feb 2017 – Present
        Lockheed Martin Aero
        •	Manage engineering design changes through the change request process for both production and retrofit.
        •	Coordinate with multiple engineering groups and downstream groups to capture all impacts for design changes.

        Mechanical Engineer II, Dec 2011 – Mar 2016
        Nuclear Logistics Inc.
        •	Select parts and design assemblies to meet client design requirements. 
        •	Write specifications and test plans to verify that components meet client requirements. 
        •	Design and manage test setups for a wide variety of electrical and mechanical components.

        Projects and Accomplishments	•	Created a metric to track Engineering Change Proposals that drive potential aircraft groundings.
        •	Automated data collection to manage and improve the Engineering Change Proposal backlog.
        •	Initiated a process improvement for Off-board changes to improve affordability and reduce span times.
        •	Created an automated tool that can track late Tech Docs for the proposal process.
        •	Created a tool that tracks aircraft that were missed between production and retrofit changes.
        •	Created a tool that can automatically generate exception reports to track Change Requests that have been delayed in the Change Process.
        •	Trained multiple new engineers on the commercial grade dedication process.
        •	Sourced, tested, documented, and shipped over 100 parts for a single project totaling over $300,000.
        •	Managed a project with over 1,500 parts, tracking within the facility and the documentation process.
        •	Presented multi-day training to clients on engineering processes.
        •	Wrote several guide documents for the engineering department to increase document quality and efficiency.
        •	Wrote several test and report templates for new engineers.
        Education	Bachelor of Science in Mechanical Engineering, 2014
        The University of Texas at Arlington
        Courses covered Thermodynamics, Heat Transfer, Fluids, Mechanics of Materials, Manufacturing, Machine Design, Control systems, Statics, Dynamics, Pro-E Solid Modeling, Circuits, and Ansys finite element analysis.

        Skills	•	PDM
        •	Microsoft Word and Excel
        •	Microsoft Access
        •	Data processing
        •	Technical writing
        •	Writing test instructions
        •	Writing stress analysis reports
        •	Writing circuit analysis reports
        •	Writing procurement specifications
        •	Project management
        •	Programming languages: C, C++, Python, VBA, SQL
        """
    job_description = """Description

        The Senior Data Scientist is responsible for defining, building, and improving statistical models to improve business processes and outcomes in one or more healthcare domains such as Clinical, Enrollment, Claims, and Finance. As part of the broader analytics team, Data Scientist will gather and analyze data to solve and address complex business problems and evaluate scenarios to make predictions on future outcomes and work with the business to communicate and support decision-making. This position requires strong analytical skills and experience in analytic methods including multivariate regressions, hierarchical linear models, regression trees, clustering methods and other complex statistical techniques.

        Duties & Responsibilities:

        â€¢ Develops advanced statistical models to predict, quantify or forecast various operational and performance metrics in multiple healthcare domains
        â€¢ Investigates, recommends, and initiates acquisition of new data resources from internal and external sources
        â€¢ Works with multiple teams to support data collection, integration, and retention requirements based on business needs
        â€¢ Identifies critical and emerging technologies that will support and extend quantitative analytic capabilities
        â€¢ Collaborates with business subject matter experts to select relevant sources of information
        â€¢ Develops expertise with multiple machine learning algorithms and data science techniques, such as exploratory data analysis and predictive modeling, graph theory, recommender systems, text analytics and validation
        â€¢ Develops expertise with Healthfirst datasets, data repositories, and data movement processes
        â€¢ Assists on projects/requests and may lead specific tasks within the project scope
        â€¢ Prepares and manipulates data for use in development of statistical models
        â€¢ Other duties as assigned

        Minimum Qualifications:

        -Bachelor's Degree

        Preferred Qualifications:

        - Masterâ€™s degree in Computer Science or Statistics
        Familiarity with major cloud platforms such as AWS and Azure
        Healthcare Industry Experience

        Minimum Qualifications:

        -Bachelor's Degree

        Preferred Qualifications:

        - Masterâ€™s degree in Computer Science or Statistics
        Familiarity with major cloud platforms such as AWS and Azure
        Healthcare Industry Experience

        WE ARE AN EQUAL OPPORTUNITY EMPLOYER. Applicants and employees are considered for positions and are evaluated without regard to mental or physical disability, race, color, religion, gender, national origin, age, genetic information, military or veteran status, sexual orientation, marital status or any other protected Federal, State/Province or Local status unrelated to the performance of the work involved.

        If you have a disability under the Americans with Disability Act or a similar law, and want a reasonable accommodation to assist with your job search or application for employment, please contact us by sending an email to careers@Healthfirst.org or calling 212-519-1798 . In your email please include a description of the accommodation you are requesting and a description of the position for which you are applying. Only reasonable accommodation requests related to applying for a position within Healthfirst Management Services will be reviewed at the e-mail address and phone number supplied. Thank you for considering a career with Healthfirst Management Services.
        EEO Law Poster and Supplement

        ]]>"""
    grader = Grader()
    result = grader.grade_resume(resume, job_description)
    print(result)

main()