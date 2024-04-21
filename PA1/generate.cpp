#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstring>
#include <string>

using namespace std;

//#define CHECK

int main(int argc, char **argv) {
  if (argc != 4) {
    cout << "Usage: ./generate <number_count> <input_file> <mode> (0 <= n <= "
            "2147483647) "
         << endl;
    return 1;
  }

  const int n = atoi(argv[1]);
  const char *input_file = argv[2];
  const char *mode = argv[3];

  fstream input_data(input_file, ios::out | ios::binary);

  if (!input_data) {
    cout << "fail to open the file" << endl;
    return -1;
  }

  float *data = new float[n]();

  for (int i = 0; i < n; i++)
#ifdef CHECK
  {
#endif
    if (strcmp(mode, "random") == 0) {
      data[i] = 20000000.0 / (rand() % 10000000);
    }
    else if (strcmp(mode, "descending") == 0) {
      data[i] = n / 2 - i;
    }
    else if (strcmp(mode, "identical") == 0) {
      data[i] = 0;
    }
    else {
      cout << "mode should be random, descending, or identical" << endl;
      delete[] data;
      return -1;
    }
#ifdef CHECK
    cout << data[i] << " ";
  }
  cout << endl;
#endif

  if (!input_data.write(reinterpret_cast<char *>(data), n * sizeof(float))) {
    cout << "fail to write" << endl;
  }
  input_data.close();

  cout << n << " elements is generated in " << input_file << endl;

  delete[] data;

#ifdef CHECK
  float *data_1 = new float[n];

  fstream output_data(input_file, ios::in | ios::binary);
  if (!output_data) {
    cout << "fail to open the file" << endl;
    return -1;
  }
  if (!output_data.read((char *)data_1, n * sizeof(float))) {
    cout << "fail to read" << endl;
  }
  output_data.close();

  for (int i = 0; i < n; i++)
    cout << data_1[i] << " ";
  cout << endl;

  delete[] data_1;
#endif

  return 0;
}
