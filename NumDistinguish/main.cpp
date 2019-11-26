#include "sqlite3/sqlite3.h"
#include <stdio.h>
#include "NumDistinguish.h"
#include "NeuralMatrix.h"
#include "Sample.h"

bool GetData(std::vector<Sample*>& datas) {
	sqlite3* db;
	if (SQLITE_OK != sqlite3_open("resources/test.db", &db)) {
		printf(sqlite3_errmsg(db));
		return false;
	}
	sqlite3_stmt* pStmt;
	sqlite3_prepare(db, "select * from tr_data", -1, &pStmt, 0);
	while (sqlite3_step(pStmt) == SQLITE_ROW) {
		int ulImageSize = sqlite3_column_bytes(pStmt, 2);
		if (ulImageSize == 3136) {
			Sample* layer = new Sample(sqlite3_column_int(pStmt, 1), (float*)sqlite3_column_blob(pStmt, 2), ulImageSize / sizeof(float));
			datas.push_back(layer);
		}
	}
	sqlite3_finalize(pStmt);
	sqlite3_close(db);
	return true;
}

int main() {
	DigitalDistinguish model;
	model.PushLayer(16, 784, 0);
	model.PushLayer(16, 16, 0);
	model.PushLayer(10, 16, 0);
	std::vector<Sample*> datas;
	if (GetData(datas)) {
		model.StartTraining(datas, 100);
	}
	for(size_t i = 0; i < datas.size(); i++){
		delete datas[i];
	}
	printf("done..");
	return 0;
}