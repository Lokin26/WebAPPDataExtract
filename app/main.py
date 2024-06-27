from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from utils import  extract_and_store_text, question_answering

load_dotenv()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        source_type = data.get('source_type')
        source_url = data.get('source_url')

        try:
           # processed_query = process_query_with_openai(query)
           query = data.get('query')
           processed_query = query
           if not all([query, source_type, source_url]):
            return jsonify({"error": "Missing required parameters"}), 400
        except Exception as e:
            app.logger.error(f"Error processing query with Claude: {str(e)}")
            return jsonify({"error": "Error processing query"}), 500

        # Extract and store data from WEB UI OR PDF 
        try:
            extract_and_store_text(source_url, is_pdf=(source_type == 'pdf'))
        except Exception as e:
            app.logger.error(f"Error extracting and storing text: {str(e)}")
            return jsonify({"error": f"Error extracting text from {source_type}"}), 500

        # Perform Response Based on the Web UI
        try:
            answer = question_answering(processed_query)
        except Exception as e:
            app.logger.error(f"Error in question answering: {str(e)}")
            return jsonify({"error": "Error generating answer"}), 500

        return jsonify({"result": answer})

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')