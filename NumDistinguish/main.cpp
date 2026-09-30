#include "NumDistinguish.h"
#include "Sample.h"
#include "sqlite3/sqlite3.h"
#include <stdio.h>

static bool GetData(vector<Sample *> &datas, int type)
{
    sqlite3 *db = nullptr;
    if (SQLITE_OK != sqlite3_open("resources/test.db", &db))
    {
        printf("%s", sqlite3_errmsg(db));
        return false;
    }
    sqlite3_stmt *pStmt = nullptr;
    int prepareResult = SQLITE_ERROR;
    switch (type)
    {
    case 1:
        prepareResult = sqlite3_prepare(db, "select * from te_data", -1, &pStmt, 0);
        break;
    case 2:
    default:
        prepareResult = sqlite3_prepare(db, "select * from tr_data", -1, &pStmt, 0);
        break;
    }
    if (prepareResult != SQLITE_OK || pStmt == nullptr)
    {
        sqlite3_close(db);
        return false;
    }
    while (sqlite3_step(pStmt) == SQLITE_ROW)
    {
        int ulImageSize = sqlite3_column_bytes(pStmt, 2);
        if (ulImageSize == 3136)
        {
            Sample *layer = new Sample(sqlite3_column_int(pStmt, 1),
                                       (const float *) sqlite3_column_blob(pStmt, 2),
                                       ulImageSize / sizeof(float));
            datas.push_back(layer);
        }
    }
    sqlite3_finalize(pStmt);
    sqlite3_close(db);
    return true;
}

int main()
{
    vector<Sample *> datas, testDatas;
    if (!GetData(datas, 2) || !GetData(testDatas, 1))
        return 1;

    DigitalDistinguish model;
    model.PushLayer(24, 784, ReLU, DerivReLU);
    model.PushLayer(16, 24, ReLU, DerivReLU);
    model.PushLayer(10, 16, SoftMax, DerivSoftMax);

    model.Training(datas, 100);

    model.Validate(datas);

    for (auto data : datas)
        delete data;
    for (auto data : testDatas)
        delete data;

    printf("done..\n");
    return 0;
}
