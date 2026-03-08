import whisper

model = whisper.load_model("small")
result = model.transcribe(r"C:\Users\HP\Desktop\Bible\Data\Test 1.aac", language="en")
print(result["text"])