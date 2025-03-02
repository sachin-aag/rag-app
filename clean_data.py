# read in list of files from markdown_results
import os
import re
files = os.listdir("markdown_results")
# create a durectory called cleaned_markdown_results
os.makedirs("cleaned_markdown_results", exist_ok=True)

# read in each file and clean the data
for file in files:
    with open(f"markdown_results/{file}", "r") as f:
        data = f.read()
        # only keep text between "Share this post" and "Comments"
        start = data.find("Share this post")
        end = data.find("Comments")
        data = data[start+len("Share this post"):end]
        # remove all URLs and [] and ()
        data = re.sub(r'\[', '', data)
        data = re.sub(r'\]', '', data)
        data = re.sub(r'\(', '', data)
        data = re.sub(r'\)', '', data)
        data = re.sub(r'https?://[^\s]+', '', data)
        # remove ! 
        data = re.sub(r'!', '', data)
        # remove all data after "Life is so rich,"
        data = data[:data.find("Life is so rich,")]


        
        with open(f"cleaned_markdown_results/{file}", "w") as f:
            f.write(data)


