from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from config import Config
from madules import find_similar_records


app = Flask(__name__, template_folder="templates")
app.config.from_object(Config)
app.config['ENV'] = 'production'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
CORS(app)

# configure console logging only
logging.basicConfig(
    level=app.config["LOG_LEVEL"], format="%(asctime)s %(levelname)s: %(message)s")
app.logger.setLevel(app.config["LOG_LEVEL"])


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(status="ok"), 200


@app.errorhandler(404)
def not_found(e):
    # Return JSON for 404 errors since HTML templates are not provided
    return jsonify(error="Resource not found"), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.exception("Server error:")
    # Return JSON for 500 errors instead of missing template
    return jsonify(error="An internal server error occurred"), 500


@app.route("/", methods=["GET", "POST"])
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
                    n=app.config["NUM_RESULTS"],
                    collection_name=app.config["COLLECTION_NAME"]
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
                # Log exception and show generic error message
                app.logger.error("Error during search: %s", e)
                error_message = "خطا در هنگام جستجو. لطفاً بعداً دوباره تلاش کنید."
        else:
            error_message = "لطفاً عبارتی را برای جستجو وارد کنید."

    # For GET requests or initial page load, render without results
    # For POST requests, render with results or error message
    # Pass the query back to the template to display in the input box
    return render_template('index.html', query=query, results=results, error_message=error_message)


# if __name__ == "__main__":
#     app.run(debug=False, host="0.0.0.0", port=5000)
