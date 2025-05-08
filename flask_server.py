from flask import Flask, render_template, request
from madules import find_similar_records

# --- Configuration ---

COLLECTION_NAME = "jira_hadiths"
# Set the number of results to return
NUM_RESULTS = 10
# --- End Configuration ---

# Explicitly tell Flask where to find templates
app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""
    error_message = None

    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            try:
                # Use find_similar_records from madules.py
                # It returns a list of [id, distance, metadata]
                similar_records = find_similar_records(
                    query_text=query,
                    n=NUM_RESULTS,
                    collection_name=COLLECTION_NAME
                )
                # Format results for the template
                results = [
                    {
                        'id': record[0],
                        # Ensure distance is float
                        'distance': record[1] if isinstance(record[1], (int, float)) else 1.0,
                        'text_ar': record[2].get("hadithText", 'متن عربی یافت نشد'),
                        'text_fa': record[2].get("farsiTranslation", 'ترجمه فارسی یافت نشد'),
                        'source': record[2].get('source', 'منبع نامشخص'),
                        # Add 'from' field if needed by template
                        'from': record[2].get('from', 'راوی نامشخص')
                    }
                    # Ensure record structure is valid before accessing
                    for record in similar_records if isinstance(record, (list, tuple)) and len(record) > 2 and isinstance(record[2], dict)
                ]
                if not results:
                    error_message = "نتیجه‌ای برای جستجوی شما یافت نشد."

            except Exception as e:
                print(f"Error during search: {e}")
                error_message = f"خطا در هنگام جستجو: {str(e)}"
        else:
            error_message = "لطفاً عبارتی را برای جستجو وارد کنید."

    # For GET requests or initial page load, render without results
    # For POST requests, render with results or error message
    # Pass the query back to the template to display in the input box
    return render_template('index.html', query=query, results=results, error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)  # Remember to turn off debug mode for production
