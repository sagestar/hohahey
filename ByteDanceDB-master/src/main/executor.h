//采用火山模型

#pragma once

#include "optimizer.h"

namespace bydb {

//火山模型中，包含3类数据，prev指针、next指针、数据data[]
struct TupleIter {
  TupleIter(Tuple* t) : tup(t) {}
  ~TupleIter() {
    for (auto expr : values) {
      delete expr;
    }
  }

  Tuple* tup;
  std::vector<Expr*> values;
};

class BaseOperator {
 public:
  BaseOperator(Plan* plan, BaseOperator* next) : plan_(plan), next_(next) {}
  virtual ~BaseOperator() { delete next_; }

  //设定为虚函数，每个Operator调用next_.exec()调用下层Operator
  virtual bool exec(TupleIter** iter = nullptr) = 0;

  Plan* plan_;
  BaseOperator* next_;
};

class CreateOperator : public BaseOperator {
 public:
  CreateOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~CreateOperator() {}

  //重写基类虚函数
  bool exec(TupleIter** iter = nullptr) override;
};

class DropOperator : public BaseOperator {
 public:
  DropOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~DropOperator() {}
  bool exec(TupleIter** iter = nullptr) override;
};

class InsertOperator : public BaseOperator {
 public:
  InsertOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~InsertOperator() {}
  bool exec(TupleIter** iter = nullptr) override;
};

class UpdateOperator : public BaseOperator {
 public:
  UpdateOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~UpdateOperator() {}
  bool exec(TupleIter** iter = nullptr) override;
};

class DeleteOperator : public BaseOperator {
 public:
  DeleteOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~DeleteOperator() {}
  bool exec(TupleIter** iter = nullptr) override;
};

class TrxOperator : public BaseOperator {
 public:
  TrxOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~TrxOperator() {}
  bool exec(TupleIter** iter = nullptr) override;
};

class ShowOperator : public BaseOperator {
 public:
  ShowOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~ShowOperator() {}
  bool exec(TupleIter** iter = nullptr) override;
};

class SelectOperator : public BaseOperator {
 public:
  SelectOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~SelectOperator() {}
  bool exec(TupleIter** iter = nullptr) override;
};

class SeqScanOperator : public BaseOperator {
 public:
  SeqScanOperator(Plan* plan, BaseOperator* next)
      : BaseOperator(plan, next), finish(false), nextTuple_(nullptr) {}
  ~SeqScanOperator() {
    for (auto iter : tuples_) {
      delete iter;
    }
  }
  bool exec(TupleIter** iter = nullptr) override;

 private:
  bool finish;
  Tuple* nextTuple_;
  std::vector<TupleIter*> tuples_;
};

class FilterOperator : public BaseOperator {
 public:
  FilterOperator(Plan* plan, BaseOperator* next) : BaseOperator(plan, next) {}
  ~FilterOperator() {}
  bool exec(TupleIter** iter = nullptr) override;

 private:
  bool execEqualExpr(TupleIter* iter);
};

class Executor {
 public:
  Executor(Plan* plan) : planTree_(plan) {}
  ~Executor() {}
  void init();
  bool exec();

 private:
  BaseOperator* generateOperator(Plan* Plan);

  Plan* planTree_;
  BaseOperator* opTree_;
};

}  // namespace bydb