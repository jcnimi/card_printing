from waitress import serve
import web_service_face_predict
serve(web_service_face_predict.app, host='0.0.0.0', port=8080)