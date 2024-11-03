import boto3
from datetime import datetime, timedelta
import getpass
import os

def get_instance_cost(instance_type, days=1):
    """
    Fetches the cost for a specific instance type over a specified number of days.
    """
    client = boto3.client('ce', region_name='us-east-1')
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start.strftime('%Y-%m-%d'), 'End': end.strftime('%Y-%m-%d')},
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        Filter={
            'And': [
                {'Dimensions': {'Key': 'SERVICE', 'Values': ['Amazon Elastic Compute Cloud - Compute']}},
                {'Dimensions': {'Key': 'INSTANCE_TYPE', 'Values': [instance_type]}}
            ]
        }
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost

def list_running_ec2_instances():
    """
    Lists all running EC2 instances in the account with instance ID and instance type.
    """
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instances.append({'InstanceId': instance['InstanceId'], 'InstanceType': instance['InstanceType']})
    return instances

def get_sagemaker_notebook_status(notebook_instance_name):
    """
    Fetches the status of a SageMaker notebook instance.
    """
    sagemaker_client = boto3.client('sagemaker')
    response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)
    return response['NotebookInstanceStatus']

def get_notebook_instance_info(notebook_instance_name):
    """
    Fetches the status and instance type of a specific SageMaker notebook instance.
    """
    sagemaker_client = boto3.client('sagemaker')
    response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)
    return {'Status': response['NotebookInstanceStatus'], 'InstanceType': response['InstanceType']}

def get_current_costs(days=1):
    """
    Fetches the current total costs for the specified time period across all AWS services.
    """
    client = boto3.client('ce', region_name='us-east-1')
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start.strftime('%Y-%m-%d'), 'End': end.strftime('%Y-%m-%d')},
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost

def update_repo(repo_url, commit_message="Updates from Jupyter notebooks"):
    """
    Updates a GitHub repository by adding, committing, and pushing changes from the current directory.

    Parameters:
    repo_url (str): The GitHub repository URL (HTTPS format without 'https://').
    commit_message (str): The commit message to use. Default is "Updates from Jupyter notebooks".
    """
    # Prompt for GitHub username and personal access token (PAT)
    username = input("GitHub Username: ")
    token = getpass.getpass("GitHub Personal Access Token (PAT): ")

    # Configure user details
    os.system('git config --global user.name "Chris Endemann"')
    os.system('git config --global user.email "endeman@wisc.edu"')

    # Ensure we are in a git repository
    if not os.path.exists(".git"):
        print("Not a Git repository. Please initialize with `git init` and add remote origin before running this.")
        return

    # Add and commit changes
    os.system("git add .")
    os.system(f'git commit -m "{commit_message}"')

    # Pull any remote changes
    os.system("git config pull.rebase false")
    os.system(f"git pull origin main")

    # Push changes to GitHub
    github_url = f"https://{username}:{token}@{repo_url}"
    os.system(f"git push {github_url} main")
    print("Repository updated successfully.")

# Example usage of the functions in this script
if __name__ == "__main__":
    # Example: Get the cost of a specific EC2 instance type over the last day
    instance_type = 'g4dn.xlarge'  # Replace with your desired instance type
    cost = get_instance_cost(instance_type)
    print(f"Cost for {instance_type} over the last day: ${cost:.2f}")

    # Example: List all running EC2 instances with their IDs and types
    running_instances = list_running_ec2_instances()
    print("Running EC2 Instances:")
    for instance in running_instances:
        print(f"Instance ID: {instance['InstanceId']}, Instance Type: {instance['InstanceType']}")

    # Example: Check the status of a specific SageMaker notebook instance
    notebook_instance_name = 'YourNotebookInstanceName'  # Replace with your notebook instance name
    status = get_sagemaker_notebook_status(notebook_instance_name)
    print(f"SageMaker Notebook '{notebook_instance_name}' Status: {status}")

    # Example: Fetch status and instance type of a specific notebook instance
    notebook_info = get_notebook_instance_info(notebook_instance_name)
    print(f"Notebook Instance Info for '{notebook_instance_name}':")
    print(f"Status: {notebook_info['Status']}, Instance Type: {notebook_info['InstanceType']}")

    # Example: Get the total AWS cost across all services over the last 7 days
    total_cost = get_current_costs(days=7)
    print(f"Total AWS costs over the last 7 days: ${total_cost:.2f}")

    # Example: Update the GitHub repository with changes
    repo_url = 'github.com/UW-Madison-DataScience/test_AWS.git'  # Update with your repository URL
    update_repo(repo_url, commit_message="Automated commit from Jupyter notebook")
