"""
######################################################################
Email With Attachments Python Script
Coded By "The Intrigued Engineer" over a coffee
Thanks For Watching!!!
######################################################################
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import datetime
# Setup port number and server name

smtp_port = 587                 # Standard secure SMTP port
smtp_server = "smtp.gmail.com"  # Google SMTP Server

# Set up the email lists
email_from = "udjatandninja74@gmail.com"
email_list = ["ahmedsalem292001@gmail.com"]#,"amabdelgalil78@gmail.com", "youyou_tarek2001@hotmail.com"]

# Define the password (better to reference externally)
pswd = "qiyqpsyyojanxpdm" # As shown in the video this password is now dead, left in as example only


# name the email subject
subject = "Security Alert !!"

# Define the email function (dont call it email!)
def send_emails():

    for person in email_list:

        # Make the body of the email
        body = " we detect a knife with yolov8  at {}".format(datetime.datetime.now())

        # make a MIME object to define parts of the email
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = person
        msg['Subject'] = subject

        # Attach the body of the message
        msg.attach(MIMEText(body, 'plain'))

        # Define the file to attach
        filename = r"D:\4th Year\detect and website\screenShot.jpg"

        # Open the file in python as a binary
        attachment = open(filename, 'rb')  # r for read and b for binary

        # Encode as base 64
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload((attachment).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + filename)
        msg.attach(attachment_package)

        # Cast as string
        text = msg.as_string()

        # Connect with the server
        print("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls()
        TIE_server.login(email_from, pswd)
        print("Succesfully connected to server")
        print()


        # Send emails to "person" as list is iterated
        print(f"Sending email to: {person}...")
        TIE_server.sendmail(email_from, person, text)
        print(f"Email sent to: {person}")
        print()

    # Close the port
    TIE_server.quit()


# Run the function

