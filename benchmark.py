import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

gguf_url = "https://boxing-grand-webmaster-pictures.trycloudflare.com/v1/chat/completions"
vllm_url = "http://100.109.59.128:8000/v1/chat/completions"

TRANSLATE_TEMPLATE = "Translate the following text to English: {text}"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-AWQ")

SYSTEM_PROMPT = (
    "You are a translator who are translating the meeting transcript in Vietnamese into English. "
    "Translate the following Vietnamese text into natural English:\n"
    "- Do not translate or omit Vietnamese names\n"
    "- Keep all proper names unchanged.\n"
    "- Use fluent, grammatically correct English.\n"
    "- Maintain the original meaning and context.\n /no-think"
)

def demo():
  vllm_data = {
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [
      {"role": "user", "content": "Give me a short introduction to large language models."}
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "chat_template_kwargs": {"enable_thinking": False}
  }

  response = requests.post(vllm_url, json=vllm_data).json()

  print(response["choices"][0]["message"]["content"])


def make_request(content, type = 'vllm'):
    prompt = TRANSLATE_TEMPLATE.replace("{text}", content)
    data = {
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": prompt}
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "chat_template_kwargs": {"enable_thinking": False}
  }
    url = vllm_url if type == 'vllm' else gguf_url
    inference_time = 0
    try: 
      start = time.perf_counter()
      response = requests.post(url, json=data).json()
      inference_time = time.perf_counter() - start
      return {"prompt": prompt, "content": response["choices"][0]["message"]["content"], "inference_time": inference_time}
    except Exception as e:
       return {"prompt": prompt, "content": e, inference_time: inference_time}

ccu_contents = [
  "Quãng năm 2017, khi 35 tuổi, tôi bắt đầu",
  "Bộ trưởng Bộ Giáo dục và Đào tạo Nguyễn Kim Sơn đã có trả lời ý kiến của cử tri một số tỉnh gửi đến trước kỳ họp thứ 9.",
  "Ông bị tình nghi phạm tội hình sự, cụ thể là biển thủ, chiếm dụng vốn dự án và tài sản;",
  "Theo Korea Times, trước thềm nhập ngũ, Cha Eun Woo có những hoạt động chia tay đầy ý nghĩa với người hâm mộ.",
  "Từng là sinh viên và sử dụng xe buýt để di chuyển, nhiều lúc anh Cường cũng gặp phải những trạm không có băng ghế ngồi chờ. Với ông bà lớn tuổi thường sử dụng xe buýt khó có thể đứng chờ lâu nếu lỡ mất chuyến.",
  "Đó là năm 2015, Nguyễn Khánh Linh 24 tuổi, đang theo học thạc sĩ kỹ thuật về AI tại Đại học Quốc gia Singapore. Lúc này cô mới được chẩn đoán bị rối loạn lo âu, được điều trị tích cực sau nhiều năm sống với những vấn đề về tâm lý trầm trọng mà không biết mình gặp phải.",
  "Hiện tại chúng tôi chưa quan sát thấy hoạt động bán cao bất thường trên các nhóm ngành chủ chốt.",
  "Điều đáng nói đây không phải là nội dung sở trường của cô bé."
]
meeting_contents = [
  "Bây giờ nhà trường muốn nghe ý kiến phản hồi từ phụ huynh.",
  "Phục lớp mình thì ngày hôm qua giáo viên chủ nhiệm đã điều tra là lớp mình đã có.",
  "Phụ huynh cũng nhất trí cho các cháu đến trường trong cái tình hình như vậy thì em cũng đồng ý thôi Nhưng mà còn những cái phụ huynh nào mà băn khoăn như em ấy, thì em cũng muốn nghe ý kiến của các phụ huynh như thế nào",
  "Rồi mà chưa có được. Dạ, em chỉ lo lắng cái khoản đó thôi. Dạ, mặc dù em biết là học online thì cũng không có cái cái kết quả là nó không được cao và học trực tiếp ở trường. Nhưng mà với cái tình hình dịch bệnh như này thì thật sự là không yên tâm chút nào nữa.",
  "Cái bảng điểm giáo viên chủ nhiệm vừa gởi đến cho phụ huynh rồi, phụ huynh có thể mở ra xem.",
  "Bây giờ, trong ở đây, lớp mình bây giờ đã có 26 phụ huynh.",
]
contents = [
    "Mình sẽ đến trường để học trực tiếp.",
    "Bây giờ nhà trường muốn nghe ý kiến phản hồi từ phụ huynh.",
    "Phục lớp mình thì ngày hôm qua giáo viên chủ nhiệm đã điều tra là lớp mình đã có.",
    "Đã có 39 em",
    "Trực tiếp ở trường.",
    "Riêng một số em vì lý do",
    "Và thứ ba nữa là mình vào cái vấn đề học tập.",
    "Cái bảng điểm giáo viên chủ nhiệm vừa gởi đến cho phụ huynh rồi, phụ huynh có thể mở ra xem.",
    "Bây giờ, trong ở đây, lớp mình bây giờ đã có 26 phụ huynh.",
    "Muốn cái chừng nào mà tình hình cái dịch bệnh nó yên ổn một chút xíu đấy thì mình hãy nên cho các cháu đến trường thì em không có yên tâm cái khoản đến trường học chút nào hết á. Dạ, mặc dù là nhà trường, nếu như nhà trường yêu cầu là phải đến trường học, thì em cũng có thể cho các cháu đến trường được. Nhưng mà em không yên tâm.Mặc dù ở nhà em ai cũng chích hết rồi cháu thì mới chích có một mũi thôi, vì cháu đang ở quê, đang đăng ký.",
    "Rồi mà chưa có được. Dạ, em chỉ lo lắng cái khoản đó thôi. Dạ, mặc dù em biết là học online thì cũng không có cái cái kết quả là nó không được cao và học trực tiếp ở trường. Nhưng mà với cái tình hình dịch bệnh như này thì thật sự là không yên tâm chút nào nữa. Dạ em chỉ băn khoăn cái khoản đó là em muốn hỏi ý kiến của cô và các phụ huynh như thế nào nếu mà.",
    "Phụ huynh cũng nhất trí cho các cháu đến trường trong cái tình hình như vậy thì em cũng đồng ý thôi Nhưng mà còn những cái phụ huynh nào mà băn khoăn như em ấy, thì em cũng muốn nghe ý kiến của các phụ huynh như thế nào.Dạ thôi em, em chả có ý kiến gì hết. Em chỉ có băn khoăn cái đó thôi. Dạ.",
    "Xin mời ý kiến phụ huynh khác.",
    "Và vấn đề mà phụ huynh Châu Chí Tài băn khoăn cũng là vấn đề băn khoăn của tất cả phụ huynh và giáo viên, chứ của giáo viên chứ không phải chỉ có một mình phụ huynh Châu Chí Tài băn khoăn đâu.",
    "Bây giờ, xin mời ý kiến của các phụ huynh khác, để mình có cái hướng bàn bạc tốt nhất cho con em mình.",
    "Được, còn phụ huynh nào nữa không ạ?",
    "Dạ rồi, chờ tí xíu ạ.",
    "Xong mình sẽ sẽ ghi cô giáo chủ nhiệm sẽ ghi lại tất cả ý kiến phụ huynh là cô giáo chủ nhiệm từ từ giải đáp ạ, dạ.",
    "Ta xác định là mình sẽ sống chung với dịch.",
    "Các con online, các con search trên mạng là có đầy đủ là có hết trên mạng.",
    "Có hết trên mạng nên giống như.",
    "Nên cái việc mà kiến thức mà các con tiếp thu vào là không chính xác.",
    "Không chính xác, chưa kể là các con kiến thức các con mà ở trên lớp cũng các con nghe giảng đâu có biết các con nghe giảng đâu.",
    "Bởi vì nếu như bật cam hết lên màn hình thì.",
    "Mạng nặng, không thể.",
    "Nào các con không thể nào ngồi trong lớp được. Các con bị thoát ra mà nó không bật cam lên cho các con nghe, giảng bài thì không biết trong lúc đó các con làm cái gì trong dạy và rõ ràng quý vị phụ huynh cũng đi ra ngoài kiếm tiền, mình đâu có ở nhà mình. Mình kiểm soát con mình 24/24 được, không kiểm soát được. Chưa kể là quý vị phụ huynh có những quý vị là bác sĩ phải đi chống dịch, để con ở nhà mình rõ ràng ở đây.",
    "Thì phụ huynh phải là người kiểm tra xem con mình có đủ khẩu trang hay không?",
    "Xem con mình có.",
    "Nước chảy đá mòn từ từ mỗi ngày một chút một, mỗi ngày một chút một.",
    "Nhưng mà có những con không làm, không hoàn thành.",
    "Rồi bây giờ mình sẽ bắt đầu làm đề tài một của Đại.",
    "Và tình huống đầu tiên số một giải quyết tình huống số một là ba mẹ bạn gái mời bạn sau ba tuần nữa ra mắt nhà bạn gái mình nên chuẩn bị cái gì?",
    "Mình chuẩn bị nhận dạng về vấn đề.",
    "Là là người đàn ông đúng nghĩa, ba mẹ bạn gái mời qua nhà ăn cơm.",
    "Phải làm gì để đạt kết quả tốt nhất?",
    "Thu thập dữ liệu là xem địa chỉ nhà khu vực.",
    "Ở vùng núi hay vùng đồng bằng, gia đình có bao nhiêu người có sống chung với ông bà hay không?",
    "Họ hàng cô chú, cậu dì, ba mẹ là gì?",
    "Sở thích của ba mẹ?",
    "Ba mẹ muốn hình mẫu bạn trai của người con gái là như thế nào?",
    "Tiếp theo là phần ba của bạn phát.",
    "Mục mục tiêu cuối cùng anh có nghe không ạ?",
    "Xác định mục tiêu cuối cùng là tạo hình là",
    "Mục tiêu chính của mình là tạo ấn tượng cho cái buổi gặp mặt với gia đình nhà gái.",
    "Ây trình bày thêm cái mục tiêu cuối cùng á.",
    "Sao ít vậy?",
    "Chưa nghĩ ra rồi, tiếp tục và bạn Điền trình bày tiếp tục đi.",
    "À, thứ nhất là về ba mẹ hoặc họ hàng của bạn gái không thích mình làm ngành nghề nào đó."
]
one_sentence = [
   "Nên là mình phải biết được là ở đó nó xưng hô như thế nào và cái thời gian, cái thời gian để làm chi để khi mà mình sắp xếp đi tới đó nó phải đúng giờ, nó đừng có bị trễ địa điểm, địa điểm thì phải biết nó xa hay nó gần. Sẽ biết được là trong cái bữa cơm của gia đình á, nó sẽ có những người như thế nào rồi? Mình phải tìm hiểu những cái người đó để giống như là."
]

def parallel_requests(contents, type='vllm'):
  processes = []
  n_requests = len(contents)
  n_threads = n_requests
  total_output_tokens = 0

  start = time.perf_counter()
  with ThreadPoolExecutor(max_workers=n_threads) as executor:
    for content in contents:
        processes.append(executor.submit(make_request, content, type))

  generated_contents = []
  for task in as_completed(processes):
      result = task.result()
      generated_contents.append(result["content"])
  elapsed = time.perf_counter() - start
    
  for output in generated_contents:
      try:
        output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
      except Exception as e:
        output_tokens = 0
        print(f"Error encoding output: {e}")
      total_output_tokens += output_tokens
      print(output)
      print("-" * 50)

  print(f"Total generated tokens: {total_output_tokens}")
  print(f"Tokens per second (parallel): {total_output_tokens / elapsed:.2f} tokens/s")
  print("Total times to handle {} requests with {} concurrent threads: {} s".format(n_requests, n_threads, elapsed))
  print(f"Successfully handled {len(generated_contents)} requests.")

def one_request_at_once(contents, type='vllm'):
  min_len = 10000
  max_len = -1
  total_generated_tokens = 0
  total_time = 0

  for content in contents:
    result = make_request(content, type)
    total_time += result["inference_time"]
    # num_words = len(result["content"].split(" "))
    num_tokens = len(tokenizer.encode(result["content"], add_special_tokens=False))
    if num_tokens < min_len: min_len = num_tokens
    if num_tokens > max_len: max_len = num_tokens
    total_generated_tokens += num_tokens
    print(result)
    print("-"*50)
  
  avg_generated_token = total_generated_tokens/len(contents)
  print("Average generated num_tokens: {}".format(avg_generated_token))
  print("Min generated num_tokens: {}, Max generated num_tokens: {}".format(min_len, max_len))
  
  print("-"*50)
  print("Total generated num_tokens: {}".format(total_generated_tokens))
  print("Tokens per second (one by one): {:.2f} tokens/s".format(total_generated_tokens / total_time))
  print("Total times to handle {} requests one by one: {} s".format(len(contents), total_time))
  print("Average time per request: {} s".format(total_time / len(contents)))

if __name__ == "__main__":
  min_len = 10000
  max_len = -1
  avg_len = 0

  contents = contents
  for content in contents:
    # num_tokens = len(TRANSLATE_TEMPLATE.replace("{text}", content).split(" "))
    num_tokens = len(tokenizer.encode(TRANSLATE_TEMPLATE.replace("{text}", content), add_special_tokens=False))
    if num_tokens < min_len: min_len = num_tokens
    if num_tokens > max_len: max_len = num_tokens
    avg_len += num_tokens
  avg_len /= len(contents)
  print("Average prompt num_tokens: {}".format(avg_len))
  print("Min prompt num_tokens: {}, Max prompt num_tokens: {}".format(min_len, max_len))

  # one_request_at_once(contents, type='gguf')
  # one_request_at_once(contents, type='vllm')
  # parallel_requests(contents, type='vllm')
  parallel_requests(contents, type='gguf')  