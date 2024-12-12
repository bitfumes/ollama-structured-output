import json
import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from ollama import chat
from pydantic import BaseModel, EmailStr

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="templates")


class ContactForm(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str
    phone: Optional[str] = None


def parse_contact_info(text: str, image: str):
    """Use Ollama to extract structured contact information from text."""

    system_prompt = """Extract contact information from the text and return it in JSON format with these fields:
    {
        "name": "full name",
        "email": "email address",
        "subject": "subject or purpose",
        "message": "main message content",
        "phone": "phone number if present, or null"
    }
    Only include these fields and ensure the output is valid JSON."""

    model = "llama3.2-vision" if image else "llama3.2"
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if image:
        messages.append({"role": "user", "images": [image]})
    else:
        messages.append({"role": "user", "content": text})
    print(model)
    try:
        response = chat(
            model=model,
            messages=messages,
            stream=False,
            format={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "subject": {"type": "string"},
                    "message": {"type": "string"},
                    "phone": {"type": "number"}
                },
                "required": ["name", "email", "subject", "message", "phone"]
            },
        )

        # Parse the response text as JSON
        print(response.message.content)
        # contact_data = json.loads(response.message.content)
        return response.message.content

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing contact info: {str(e)}")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.post("/contact")
async def contact(request: Request):
    data = await request.form()
    message = data.get("message")
    image = data.get("image")
    image_path = None
    if image:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            # Write the uploaded file content to our temporary file
            shutil.copyfileobj(image.file, tmp)
            # Get the absolute path
            image_path = os.path.abspath(tmp.name)
    contact_data = parse_contact_info(message, image_path)
    return contact_data
