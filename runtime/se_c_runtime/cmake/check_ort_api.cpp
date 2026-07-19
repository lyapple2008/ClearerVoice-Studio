#include <onnxruntime_c_api.h>

int main() {
  const OrtApiBase* base = OrtGetApiBase();
  if (base == nullptr || base->GetApi(ORT_API_VERSION) == nullptr) return 1;
  return 0;
}
