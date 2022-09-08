import os
from django.http import JsonResponse
from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import csrf_exempt


fm = FaceMesh()
index = 1
def index(requset):
    return render(requset, "index.html", {})

@csrf_exempt
def uploadImage(request):
    result_dict = {}
    try:
        new_files = request.FILES.getlist('uplaod_file')  # 새로 업로드한 파일 리스트 저장
        fpath = 'upload_files/image/'
        os.makedirs(fpath, exist_ok=True)
        for new_file in new_files:
            file_name = new_file.name
            with open(fpath + "test.jpg", 'wb') as file:
                for chunk in new_file.chunks():
                    file.write(chunk)
            file.close()
        predict = fm.predict(fpath)
        result_dict["data"] = predict
        result_dict["result"] = "success"
    except Exception as e:
        print(e)
        result_dict["result"] = "fail"
    return JsonResponse(result_dict)