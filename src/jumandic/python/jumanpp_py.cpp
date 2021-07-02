#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "jumandic/shared/jumandic_env.h"

using namespace jumanpp;

static jumandic::JumanppExec *exec = nullptr;

static PyObject* jumanpp_init(PyObject *self, PyObject *args) {
  const char* model_file;
  if (!PyArg_ParseTuple(args, "s", &model_file))
    return nullptr;

  if (exec) {
    PyErr_SetString(PyExc_RuntimeError, "Already inited");
    return nullptr;
  }
  core::analysis::rnn::RnnInferenceConfig rnn_config{};
  rnn_config.nceBias = 5.62844432562;
  rnn_config.unkConstantTerm = -3.4748115191;
  rnn_config.unkLengthPenalty = -2.92994951022;
  rnn_config.perceptronWeight = 1;
  rnn_config.rnnWeight = 0.0176;
  jumandic::JumanppConf conf;
  conf.modelFile = model_file;
//  conf.logLevel = 5;
  conf.rnnConfig.mergeWith(rnn_config);
  exec = new jumandic::JumanppExec{conf};
  Status s = exec->init();
  if (!s.isOk()) {
    PyErr_SetString(PyExc_RuntimeError, s.message().str().c_str());
    delete exec;
    exec = nullptr;
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyObject* jumanpp_analyze(PyObject *self, PyObject *args) {
  const char* input;
  if (!PyArg_ParseTuple(args, "s", &input))
    return nullptr;
  if (!exec) {
    PyErr_SetString(PyExc_RuntimeError, "Not inited");
    return nullptr;
  }
  Status s = exec->analyze(std::string(input));
  if (!s.isOk()) {
    PyErr_SetString(PyExc_RuntimeError, s.message().str().c_str());
    return nullptr;
  }
  std::string result = exec->format()->result().str();
  return PyUnicode_FromString(result.c_str());
}

static PyMethodDef jumanpp_methods[] = {
  {
    "init",
    jumanpp_init,
    METH_VARARGS,
    nullptr
  },
  {
    "analyze",
    jumanpp_analyze,
    METH_VARARGS,
    nullptr
  },
  {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef jumanpp_module = {
    PyModuleDef_HEAD_INIT,
    "jumanpp",
    nullptr,
    -1,
    jumanpp_methods
};

PyMODINIT_FUNC PyInit_jumanpp() {
  return PyModule_Create(&jumanpp_module);
}