import os
import warnings

# Disabilita warnings prima di qualsiasi import
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from elia.server import create_app
app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=app.config["PORT"], debug=app.config["DEBUG"])
