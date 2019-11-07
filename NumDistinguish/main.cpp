#include "sqlite3/sqlite3.h"
#include <stdio.h>
#include "NumDistinguish.h"
#include "NeuralLayer.h"

static int callback(void* NotUsed, int argc, char** argv, char** azColName) {

	for (int i = 0; i < argc; i++) {
		printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
	}
	printf("\n");
	return 0;
}

bool GetData(std::vector<RawData>& datas) {
	sqlite3* db;
	if (SQLITE_OK != sqlite3_open("C:\\project\\NeuralNetWorks\\test.db", &db)) {
		printf(sqlite3_errmsg(db));
		return false;
	}
	sqlite3_stmt* pStmt;
	sqlite3_prepare(db, "select * from tr_data", -1, &pStmt, 0);
	while (sqlite3_step(pStmt) == SQLITE_ROW) {
		int ulImageSize = sqlite3_column_bytes(pStmt, 2);
		if (ulImageSize == 3136) {
			datas.push_back(RawData(sqlite3_column_int(pStmt, 1), (float*)sqlite3_column_blob(pStmt, 2), ulImageSize));
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
	std::vector<RawData> datas;
	if (GetData(datas)) {
		model.StartTraining(datas);
	}
	printf("done..");
	return 0;
}