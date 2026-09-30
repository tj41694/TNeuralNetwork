#include "Sample.h"
#include <cstdlib> // Header file needed to use srand and rand
#include <ctime>   // Header file needed to use time

using namespace std;

TnVector TnVector::operator*(const TnLayer &layer) const
{
    TnVector newVector;
    newVector.resize(layer.row(), 0);
    for (int r = 0; r < layer.row(); r++)
    {
        for (int c = 0; c < layer.col(); c++)
        {
            newVector[r] += at(c) * layer.matrix[r][c];
        }
        newVector[r] = newVector[r] + layer.bias[r];
    }
    return newVector;
}

Sample::Sample()
{
    m_realValue = 0;
}

Sample::Sample(const Sample &sample_) : TnVector(sample_)
{
    m_realValue = sample_.m_realValue;
}

Sample::Sample(int num_, const float *data, unsigned int floatCount)
{
    resize(floatCount);
    for (unsigned int i = 0; i < floatCount; i++)
    {
        (*this)[i] = data[i];
    }
    m_realValue = num_;
}

Sample Sample::operator*(const TnLayer &matrix) const
{
    Sample result(*this);
    static_cast<TnVector &>(result) = TnVector::operator*(matrix);
    return result;
}

void Sample::MatrixMultiply(const TnLayer &neuralMat, ActiveFunc func)
{

    vector<double> *lastActiveLayer;

    m_activeLayers.size() == 0 ? lastActiveLayer = this
                               : lastActiveLayer = &m_activeLayers[m_activeLayers.size() - 1].a;

    if ((int) lastActiveLayer->size() != neuralMat.col())
    {
        printf("err.. Dimension not match..\n");
        return;
    }

    SampleLayer layer;
    layer.activeFunc = func;
    layer.z = TnVector::operator*(neuralMat);
    layer.a = layer.z;
    neuralMat.Active(layer.a);
    m_activeLayers.push_back(layer);
}

double Sample::GetCostValue(CostFunc func)
{
    double costVal = 0;
    switch (func)
    {
    case CostFunc::CrossEntropy:
        costVal = -log((*this)[m_realValue]);
        break;
    case CostFunc::MeanSquare:
    default:
        for (int i = 0; i < (int) (*this).size(); i++)
        {
            if (i == m_realValue)
            {
                costVal += ((*this)[i] - 1.0) * ((*this)[i] - 1.0);
            }
            else
            {
                costVal += (*this)[i] * (double) (*this)[i];
            }
        }
        break;
    }
    return costVal;
}

TnLayer::TnLayer(int row_, int colum_, ActiveFuncPtr actFunc) : activeFunc(actFunc)
{
    static bool initial = false;
    if (!initial)
    {
        srand((unsigned int) time(0));
        initial = true;
    }
    bias.resize(row_);
    for (int r = 0; r < row_; r++)
    {
        vector<double> row;
        row.resize(colum_);
        for (int c = 0; c < colum_; c++)
        {
            row[c] = rand() * 2.0 / RAND_MAX - 1.0;
        }
        matrix.emplace_back(row);
        bias[r] = rand() * 2.0 / RAND_MAX - 1.0;
    }
}

TnLayer::TnLayer(const TnLayer &neural) : activeFunc(neural.activeFunc)
{
    bias.resize(neural.bias.size(), 0);
    for (int r = 0; r < neural.row(); r++)
    {
        vector<double> row;
        row.resize(neural.col(), 0);
        matrix.emplace_back(row);
    }
}
int TnLayer::row() const
{
    return (int) matrix.size();
}
int TnLayer::col() const
{
    if (matrix.empty())
        return 0;
    return (int) matrix.front().size();
}

void TnLayer::Active(TnVector &vec) const
{
    if(activeFunc)
        activeFunc(vec);
}
