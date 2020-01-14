#pragma once
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <assert.h>
#include <omp.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/timer.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <glog/logging.h>
#include <Eigen/Dense>
#include "../SfM/sfm/SfM.h"

#define PI 3.1415926535897932384626433832795
#define FD2R(d) ((d) * (PI / 180.f)) // �Ƕ�ת����
#define FR2D(r) ((r) * (180.f / PI)) // ����ת�Ƕ�
#define DECLARE_NO_INDEX(...) std::numeric_limits<__VA_ARGS__>::max()
#define NO_ID DECLARE_NO_INDEX(uint32_t)
#define ASSERT(exp) assert(exp)
// number
#define FRONT 0
#define BACK 1
#define PLANAR 2
#define CLIPPED 3
#define CULLED 4
#define VISIBLE 5
#define ZERO_TOLERANCE (1e-7)
#define INV_ZERO (1e+14)

using namespace std;

namespace boost
{
	namespace serialization
	{
		// Point3f���л��洢
		template <class Archive>
		void serialize(Archive &ar, cv::Point3f &pt, const unsigned int)
		{
			ar &pt.x;
			ar &pt.y;
			ar &pt.z;
		}

		// Point3d���л��洢
		template <class Archive>
		void serialize(Archive &ar, cv::Point3d &pt, const unsigned int)
		{
			ar &pt.x;
			ar &pt.y;
			ar &pt.z;
		}

		// Point3i���л��洢
		template <class Archive>
		void serialize(Archive &ar, cv::Point3i &pt, const unsigned int)
		{
			ar &pt.x;
			ar &pt.y;
			ar &pt.z;
		}

		// Matx33f���л��洢
		template <class Archive>
		void serialize(Archive &ar, cv::Matx33f &mat, const unsigned int)
		{
			ar &mat(0, 0);
			ar &mat(0, 1);
			ar &mat(0, 2);
			ar &mat(1, 0);
			ar &mat(1, 1);
			ar &mat(1, 2);
			ar &mat(2, 0);
			ar &mat(2, 1);
			ar &mat(2, 2);
		}

		// Matx34f���л��洢
		template <class Archive>
		void serialize(Archive &ar, cv::Matx34f &mat, const unsigned int)
		{
			ar &mat(0, 0);
			ar &mat(0, 1);
			ar &mat(0, 2);
			ar &mat(0, 3);
			ar &mat(1, 0);
			ar &mat(1, 1);
			ar &mat(1, 2);
			ar &mat(1, 3);
			ar &mat(2, 0);
			ar &mat(2, 1);
			ar &mat(2, 2);
			ar &mat(2, 3);
		}
	} // namespace serialization
} // namespace boost

namespace REC3D
{
	// ͼ����
	struct NeighborInfo
	{
		unsigned ID;
		cv::Matx33f Hl;
		cv::Point3f Hm;
		cv::Matx33f Hr;
	};

	class Image
	{
	public:
		string name = "";					  // ͼ���ļ���
		int ID = -1;						  // ͼ��ID
		int group = 0;						  // ���������
		bool valid = 0;						  // ͼ���Ƿ���Ч
		unsigned width = 0, height = 0;		  // ͼ��ߴ�
		float k1 = 0, k2 = 0;				  // ����ϵ��
		float k3 = 0, k4 = 0, k5 = 0;		  // ����ϵ��
		cv::Point3d gps;					  // GPS
		cv::Point3d gpsLocal;				  // GPS local
		cv::Mat imageRGB;					  // ͼ���ɫ����
		cv::Mat imageGray;					  // ͼ��Ҷ�����
		cv::Matx33f K = cv::Matx33f::zeros(); // �ڲ�������
		cv::Matx33f R = cv::Matx33f::zeros(); // ��ת����
		cv::Point3f C;						  // �������
		cv::Point3f T;						  // ���ƽ������ T=-RC
		cv::Matx34f P;						  // ͶӰ����

		float scale = 1.f;				// ���ű���
		vector<NeighborInfo> neighbors; // ����ͼ����
		vector<cv::Point3f> points;		// ����ɼ�ϡ���
		vector<cv::Point3f> pointsLidar;// �����״��
		cv::Mat depthMap;				// ���ͼ		
		cv::Mat confMap;				// һ����ͼ
		cv::Mat normalMap;				// ����ͼ
		cv::Mat maskFilter;			    // ���ͼ����mask
		int num = 0;					// ��Ч�������
		float dMin, dMax;				// ���ֵ��Χ

		// Image���л��洢�����ֳ�Ա��
		template <class Archive>
		void serialize(Archive &ar, const unsigned int version)
		{
			ar &name;
			ar &ID;
			ar &group;
			ar &valid;
			ar &width;
			ar &height;
			ar &k1;
			ar &k2;
			ar &k3;
			ar &k4;
			ar &k5;
			ar &gps;
			ar &gpsLocal;
			ar &K;
			ar &R;
			ar &C;
			ar &T;
			ar &P;
		}

		// �ж϶�ά���Ƿ���ͼ����
		inline bool IsInside(const cv::Point2f &pt) const
		{
			return pt.x >= 0 && pt.y >= 0 && pt.x + 1 < width && pt.y + 1 < height;
		}
		inline bool IsInside(const cv::Point2i &pt) const
		{
			return pt.x >= 0 && pt.y >= 0 && pt.x < width && pt.y < height;
		}

		// ͼ���˫���Բ�ֵ����
		inline float Sample(const cv::Point2f &pt) const
		{
			const int lx = (int)pt.x;
			const int ly = (int)pt.y;
			const float x = pt.x - lx;
			const float y = pt.y - ly;
			const float x1 = 1.f - x;
			const float y1 = 1.f - y;
			const cv::Point2i pt1(lx, ly);
			const cv::Point2i pt2(lx + 1, ly);
			const cv::Point2i pt3(lx, ly + 1);
			const cv::Point2i pt4(lx + 1, ly + 1);
			return (imageGray.at<float>(pt1) * x1 + imageGray.at<float>(pt2) * x) * y1 + (imageGray.at<float>(pt3) * x1 + imageGray.at<float>(pt4) * x) * y;
		}

		// ��������ϵ�ռ��ͶӰ���
		inline float PointDepth(const cv::Point3f &X) const
		{
			return P(2, 0) * X.x + P(2, 1) * X.y + P(2, 2) * X.z + P(2, 3);
		}

		// ��������ϵ�ռ��ͶӰ��ͼ��
		inline cv::Point2f ProjectPointP(const cv::Point3f &X) const
		{
			cv::Point3f q(P(0, 0) * X.x + P(0, 1) * X.y + P(0, 2) * X.z + P(0, 3),
						P(1, 0) * X.x + P(1, 1) * X.y + P(1, 2) * X.z + P(1, 3),
						P(2, 0) * X.x + P(2, 1) * X.y + P(2, 2) * X.z + P(2, 3));
			return cv::Point2f(q.x / q.z, q.y / q.z);
		}

		// ��������ϵ�ռ��ͶӰ��ͼ��(��������)
		inline cv::Point2i ProjectPointPi(const cv::Point3f &X) const
		{
			cv::Point3f q(P(0, 0) * X.x + P(0, 1) * X.y + P(0, 2) * X.z + P(0, 3),
						P(1, 0) * X.x + P(1, 1) * X.y + P(1, 2) * X.z + P(1, 3),
						P(2, 0) * X.x + P(2, 1) * X.y + P(2, 2) * X.z + P(2, 3));
			return cv::Point2i(round(q.x / q.z), round(q.y / q.z));
		}

		// ��������ϵ�ռ��ͶӰ��ͼ��(�������)
		inline cv::Point3f ProjectPointP3(const cv::Point3f &X) const
		{
			return cv::Point3f(P(0, 0) * X.x + P(0, 1) * X.y + P(0, 2) * X.z + P(0, 3),
							P(1, 0) * X.x + P(1, 1) * X.y + P(1, 2) * X.z + P(1, 3),
							P(2, 0) * X.x + P(2, 1) * X.y + P(2, 2) * X.z + P(2, 3));
		}

		// ��������ϵ�ռ��ת�����������ϵ
		inline cv::Point3f TransformPointW2C(const cv::Point3f &X) const
		{
			return R * (X - C);
		}

		// �������ϵ�ռ��ͶӰ��ͼ��
		inline cv::Point2f TransformPointC2I(const cv::Point3f &X) const
		{
			return cv::Point2f(K(0, 2) + K(0, 0) * X.x / X.z,
							K(1, 2) + K(1, 1) * X.y / X.z);
		}

		// �������ϵ�ռ��ͶӰ��ͼ��(��������)
		inline cv::Point2i TransformPointC2Ii(const cv::Point3f &X) const
		{
			return cv::Point2i(round(K(0, 2) + K(0, 0) * X.x / X.z),
							round(K(1, 2) + K(1, 1) * X.y / X.z));
		}

		// ͼ������ϵ��ת�����������ϵ
		inline cv::Point3f TransformPointI2C(const cv::Point3f &X) const
		{
			return cv::Point3f((X.x - K(0, 2)) * X.z / K(0, 0),
							(X.y - K(1, 2)) * X.z / K(1, 1),
							X.z);
		}

		// �������ϵ�ռ��ת������������ϵ
		inline cv::Point3f TransformPointC2W(const cv::Point3f &X) const
		{
			return R.t() * X + C;
		}

		// ͼ������ϵ��ת������������ϵ
		inline cv::Point3f TransformPointI2W(const cv::Point3f &X) const
		{
			return TransformPointC2W(TransformPointI2C(X));
		}
	}; // ͼ����

	typedef vector<Image> ImageArr;
	typedef pair<int, int> ImagePairId;
	typedef vector<ImagePairId> ImagePairIds;

	// ������
	class PointCloud
	{
	public:
		typedef vector<cv::Point3f> PointArr, NormalArr;
		typedef vector<cv::Point3i> ColorArr;
		typedef vector<unsigned> ViewArr, LabelArr;
		typedef vector<ViewArr> PointViewArr;
		typedef vector<float> WeightArr;
		typedef vector<WeightArr> PointWeightArr;

	public:
		PointArr points;			 // ��ά��������
		PointViewArr pointViews;	 // ���ƿ���ͼ�����
		PointViewArr pointTracks;	 // ���ƿ���ͼ�����������
		PointWeightArr pointWeights; // ���ƿ���ͼ��Ȩ��
		ColorArr colors;			 // ������ɫ
		NormalArr normals;			 // ���Ʒ���
		LabelArr labels;             // �������

		// PointCloud���л��洢
		template <class Archive>
		void serialize(Archive &ar, const unsigned int version)
		{
			ar &points;
			ar &pointViews;
			ar &pointTracks;
			ar &colors;
			ar &normals;
			ar &labels;
		}

	public:
		// ȥ�����������ά��
		inline void RemovePoint(unsigned long long idx)
		{
			if (!points.empty())
				points.erase(points.begin() + idx);
			if (!pointViews.empty())
				pointViews.erase(pointViews.begin() + idx);
			if (!pointTracks.empty())
				pointTracks.erase(pointTracks.begin() + idx);
			if (!colors.empty())
				colors.erase(colors.begin() + idx);
			if (!normals.empty())
				normals.erase(normals.begin() + idx);
			if (!pointWeights.empty())
				pointWeights.erase(pointWeights.begin() + idx);
			if (!labels.empty())
				labels.erase(labels.begin() + idx);
		}

		// ��յ���
		inline void Clear()
		{
			points.clear();
			pointViews.clear();
			pointTracks.clear();
			colors.clear();
			normals.clear();
			pointWeights.clear();
			labels.clear();
		}

		inline PointCloud()
		{
			Clear();
		}
	}; // ������

	//������
	template <typename TYPE>
	class TPoint3 : public cv::Point3_<TYPE>
	{
	public:
		// pointer to the first element access
		inline const TYPE *ptr() const { return &this->x; }
		inline TYPE *ptr() { return &this->x; }

		// 1D element access
		inline const TYPE &operator[](size_t i) const
		{
			ASSERT(i >= 0 && i < 3);
			return ptr()[i];
		}
		inline TYPE &operator[](size_t i)
		{
			ASSERT(i >= 0 && i < 3);
			return ptr()[i];
		}

	}; //������

	// Basic axis-aligned bounding-box class �߽����
	template <typename TYPE, int DIMS>
	class TAABB
	{
		//STATIC_ASSERT(DIMS > 0 && DIMS <= 3);

	public:
		typedef TYPE Type;
		typedef Eigen::Matrix<TYPE, DIMS, 1> POINT;
		typedef Eigen::Matrix<TYPE, DIMS, DIMS, Eigen::RowMajor> MATRIX;

		POINT ptMin, ptMax; // box extreme points

		inline TAABB() {}
		inline TAABB(const POINT &_pt)
			: ptMin(_pt), ptMax(_pt) //�Ƿ���ȷ��ֵ������������������������������
		{
		}

		// ��������Ե����߿�
		void Insert(const POINT &pt)
		{
			if (ptMin[0] > pt[0])
				ptMin[0] = pt[0];
			else if (ptMax[0] < pt[0])
				ptMax[0] = pt[0];

			if (DIMS > 1)
			{
				if (ptMin[1] > pt[1])
					ptMin[1] = pt[1];
				else if (ptMax[1] < pt[1])
					ptMax[1] = pt[1];
			}

			if (DIMS > 2)
			{
				if (ptMin[2] > pt[2])
					ptMin[2] = pt[2];
				else if (ptMax[2] < pt[2])
					ptMax[2] = pt[2];
			}
		}

	}; // class TAABB

	// Basic hyper-plane class ƽ���ࣨ����;��룩
	// (plane represented in Hessian Normal Form: n.x+d=0 <=> ax+by+cz+d=0)
	template <typename TYPE, int DIMS = 3>
	class TPlane
	{
		//STATIC_ASSERT(DIMS > 0 && DIMS <= 3);

	public:
		typedef Eigen::Matrix<TYPE, DIMS, 1> VECTOR;
		typedef Eigen::Matrix<TYPE, DIMS, 1> POINT;
		typedef TAABB<TYPE, DIMS> AABB;

		VECTOR m_vN; // plane normal vector
		TYPE m_fD;   // distance to origin ƽ�浽ԭ��ľ���

		inline TPlane() {}
		inline TPlane(const POINT &p0, const POINT &p1, const POINT &p2)
		{
			Set(p0, p1, p2);
		}
		
		inline TPlane(const cv::Point3_<TYPE> &a, const cv::Point3_<TYPE> &b, const cv::Point3_<TYPE> &c)
		{
			POINT p0(a.x, a.y, a.z);
			POINT p1(b.x, b.y, b.z);
			POINT p2(c.x, c.y, c.z);
			Set(p0, p1, p2);
		}

		//PONIT ����
		inline void Set(const POINT &p0, const POINT &p1, const POINT &p2)
		{
			const VECTOR vcEdge1 = p1 - p0;
			const VECTOR vcEdge2 = p2 - p0;
			m_vN = vcEdge1.cross(vcEdge2).normalized(); // ��λ������
			m_fD = -m_vN.dot(p0);
		}

		// Calculate distance to point. Plane normal must be normalized.
		inline double Distance(const POINT &p) const
		{
			return m_vN.dot(p) + m_fD;
		}

	}; // class TPlane

	// Basic frustum class  ��׵����
	// (represented as 6 planes oriented toward outside the frustum volume)
	template <typename TYPE, int DIMS = 6>
	class TFrustum
	{
		//STATIC_ASSERT(DIMS > 0 && DIMS <= 6);

	public:
		typedef Eigen::Matrix<TYPE, 4, 4, Eigen::RowMajor> MATRIX4x4;
		typedef Eigen::Matrix<TYPE, 3, 4, Eigen::RowMajor> MATRIX3x4;
		typedef Eigen::Matrix<TYPE, 4, 1> VECTOR4;
		typedef Eigen::Matrix<TYPE, 3, 1> VECTOR;
		typedef Eigen::Matrix<TYPE, 3, 1> POINT;
		typedef TPlane<TYPE, 3> PLANE;
		typedef TAABB<TYPE, 3> AABB;

		PLANE m_planes[DIMS]; // left, right, top, bottom, near and far planes

		inline TFrustum() {}
		inline TFrustum(const cv::Matx34f &m, TYPE width, TYPE height, TYPE n = TYPE(0.0001), TYPE f = TYPE(1000))
		{
			MATRIX3x4 m1; //(m(0, 0), m(0, 1), m(0, 2), m(0, 3); m(1, 0), m(1, 1), m(1, 2), m(1, 3); m(2, 0), m(2, 1), m(2, 2), m(2, 3));
			m1 << m(0, 0), m(0, 1), m(0, 2), m(0, 3),
				m(1, 0), m(1, 1), m(1, 2), m(1, 3),
				m(2, 0), m(2, 1), m(2, 2), m(2, 3);
			MATRIX4x4 M(MATRIX4x4::Identity());
			M.template topLeftCorner<3, 4>() = m1;
			//����ֵ
			const VECTOR4 ltn(0, 0, n, 1), rtn(width * n, 0, n, 1), lbn(0, height * n, n, 1), rbn(width * n, height * n, n, 1);
			const VECTOR4 ltf(0, 0, f, 1), rtf(width * f, 0, f, 1), lbf(0, height * f, f, 1), rbf(width * f, height * f, f, 1);
			const MATRIX4x4 inv(M.inverse()); //�������
			const VECTOR4 ltn3D(inv * ltn), rtn3D(inv * rtn), lbn3D(inv * lbn), rbn3D(inv * rbn);
			const VECTOR4 ltf3D(inv * ltf), rtf3D(inv * rtf), lbf3D(inv * lbf), rbf3D(inv * rbf);
			m_planes[0].Set(ltn3D.template topRows<3>(), ltf3D.template topRows<3>(), lbf3D.template topRows<3>());
			if (DIMS > 1)
				m_planes[1].Set(rtn3D.template topRows<3>(), rbf3D.template topRows<3>(), rtf3D.template topRows<3>());
			if (DIMS > 2)
				m_planes[2].Set(ltn3D.template topRows<3>(), rtf3D.template topRows<3>(), ltf3D.template topRows<3>());
			if (DIMS > 3)
				m_planes[3].Set(lbn3D.template topRows<3>(), lbf3D.template topRows<3>(), rbf3D.template topRows<3>());
			if (DIMS > 4)
				m_planes[4].Set(ltn3D.template topRows<3>(), lbn3D.template topRows<3>(), rbn3D.template topRows<3>());
			if (DIMS > 5)
				m_planes[5].Set(ltf3D.template topRows<3>(), rtf3D.template topRows<3>(), rbf3D.template topRows<3>());
		}

		unsigned int Classify(const AABB &aabb) const
		{
			bool bIntersects = false;

			// find and test extreme points
			for (int i = 0; i < DIMS; ++i)
			{
				const PLANE &plane = m_planes[i];
				POINT ptPlaneMin, ptPlaneMax;

				// x coordinate
				if (plane.m_vN(0) >= TYPE(0))
				{
					ptPlaneMin(0) = aabb.ptMin(0);
					ptPlaneMax(0) = aabb.ptMax(0);
				}
				else
				{
					ptPlaneMin(0) = aabb.ptMax(0);
					ptPlaneMax(0) = aabb.ptMin(0);
				}
				// y coordinate
				if (plane.m_vN(1) >= TYPE(0))
				{
					ptPlaneMin(1) = aabb.ptMin(1);
					ptPlaneMax(1) = aabb.ptMax(1);
				}
				else
				{
					ptPlaneMin(1) = aabb.ptMax(1);
					ptPlaneMax(1) = aabb.ptMin(1);
				}
				// z coordinate
				if (plane.m_vN(2) >= TYPE(0))
				{
					ptPlaneMin(2) = aabb.ptMin(2);
					ptPlaneMax(2) = aabb.ptMax(2);
				}
				else
				{
					ptPlaneMin(2) = aabb.ptMax(2);
					ptPlaneMax(2) = aabb.ptMin(2);
				}

				if (plane.m_vN.dot(ptPlaneMin) > -plane.m_fD)
					return CULLED;

				if (plane.m_vN.dot(ptPlaneMax) >= -plane.m_fD)
					bIntersects = true;
			} // for

			if (bIntersects)
				return CLIPPED;
			return VISIBLE;
		}

	}; // class TPlane

	// Basic ray class  �����ࣨԴ��͵�λ��������camera--point ray
	template <typename TYPE>
	class TRay
	{
		// typedef double TYPE;

	public:
		typedef Eigen::Matrix<TYPE, 4, 4, Eigen::RowMajor> MATRIX;
		typedef Eigen::Matrix<TYPE, 3, 1> VECTOR;
		typedef Eigen::Matrix<TYPE, 3, 1> POINT;

		VECTOR m_vDir; // ray direction (normalized)
		POINT m_pOrig; // ray origin

		// �������ߺ�ƽ��Ľ���
		const inline cv::Point3_<TYPE> Intersects(TPlane<TYPE> plane)
		{
			POINT point( m_pOrig + (m_vDir * IntersectsDist(plane)));
			return cv::Point3_<TYPE>(point[0],point[1],point[2]);
		}

		TYPE IntersectsDist(TPlane<TYPE> plane)
		{
			const TYPE Vd(plane.m_vN.dot(m_vDir)); //dot ��ʾ���
			const TYPE Vo(-plane.Distance(m_pOrig));
			return (Vd == TYPE(0) ? INV_ZERO : Vo / Vd);
		}

		inline TRay() {}
		inline TRay(const cv::Point3f &pOrig, const cv::Point3f &vDir)
			: m_vDir(vDir.x, vDir.y, vDir.z), m_pOrig(pOrig.x, pOrig.y, pOrig.z)
		{
			// ASSERT(abs(m_vDir.squaredNorm() - (double)1) < ZERO_TOLERANCE);
		} // constructors

	}; // class TRay

	// ������
	class Mesh
	{
	public:
		typedef uint32_t VIndex;
		//typedef vector<Point3f> VertexArr;

		// ��face���������������
		class Face
		{
		public:
			int areanum;				  // ��¼���ɸ����������ţ����ھ���ʹ��
			REC3D::TPoint3<VIndex> Facer; // ��¼������
		public:
			Face() { areanum = -1; }
		};
		//typedef TPoint3<VIndex> Face;
		typedef vector<Face> FaceArr;

		typedef float Type;
		typedef cv::Point3_<Type> Vertex;
		typedef uint32_t FIndex;
		typedef vector<Vertex> VertexArr;
		typedef vector<VIndex> VertexIdxArr;
		typedef vector<FIndex> FaceIdxArr;
		typedef vector<VertexIdxArr> VertexVerticesArr;
		typedef vector<FaceIdxArr> VertexFacesArr;

		typedef cv::Point3_<Type> Normal;
		typedef vector<Normal> NormalArr;
		typedef vector<bool> BoolArr;

	public:
		VertexArr vertices;
		FaceArr faces;

		NormalArr vertexNormals;		  // for each vertex, the normal to the surface in that point (optional)
		VertexVerticesArr vertexVertices; // for each vertex, the list of adjacent vertices (optional)
		VertexFacesArr vertexFaces;		  // for each vertex, the list of faces containing it (optional)
		BoolArr vertexBoundary;			  // for each vertex, stores if it is at the boundary or not (optional)
		NormalArr faceNormals;			  // for each face, the normal to it (optional)

	public:
		// ���������л��洢�����֣�
		template <class Archive>
		void serialize(Archive &ar, const unsigned int version)
		{
			ar &vertices;
			ar &faces;
			//ar & vertexNormals;
			//ar & vertexVertices;
			//ar & vertexFaces;
			//ar & vertexBoundary;
			//ar & faceNormals;
		}

		// �������������
		bool FixNonManifold();	 // ����ͨ���
		void ListIncidenteFaces(); //�г�����ڽӵ���

		// ��ȡface�Ķ���
		static inline uint32_t FindVertex(const Face &f, VIndex v)
		{
			for (uint32_t i = 0; i < 3; ++i)
				if (f.Facer[i] == v)
					return i;
			return NO_ID;
		}
		static inline VIndex GetVertex(const Face &f, VIndex v)
		{
			const uint32_t idx(FindVertex(f, v));
			ASSERT(idx != NO_ID);
			return f.Facer[idx];
		}
		static inline VIndex &GetVertex(Face &f, VIndex v)
		{
			const uint32_t idx(FindVertex(f, v));
			if (idx != NO_ID)
				return f.Facer[idx];
		}

		// ��������smooth��close holes �ȣ�
		void Clean(float fDecimate = 1.f, float fSpurious = 10.f, bool bRemoveSpikes = true, unsigned nCloseHoles = 30, unsigned nSmoothMesh = 2, bool bLastClean = true);
		void Clean2(float fDecimate = 1.f, float fSpurious = 10.f, bool bRemoveSpikes = true, unsigned nCloseHoles = 30, unsigned nSmoothMesh = 2, bool bLastClean = true);
		void Removetolongedges(double percent, double scale);
		void Smooth(int percent);
		void Decimate(double fDecimate);
		void RemoveNomanifoldvertex();

		// mesh���
		void Release()
		{
			const VIndex v0 = 2;
			vertices.clear();
			faces.clear();
		}

	}; // ������

	// ���ƻ�������
	class voxel
	{
	public:
		vector<long> Pointindex; // ��Ÿ�voxel�ڵ�����
		vector<float> Z;		 // ��ŵ��Z�����꣬������㷽��
		double variance;		 // ����
		double score;
		double s;
		double t;
		int bon;			  // ����Ƿ��Ǳ߽磬-1��ʾ�Ǳ߽�
		int tag;			  // ����Ƿ����ڸ�voxel,��ʼ��Ϊ-1
		int time;			  // ��Ҫ���ظ�����Ĵ���������������йأ������Ƿ��б�Ҫ��
		set<int> local_order; // ��Ű�����voxel�������(��locales�е��±��)������˳�������Ҷ�Ӧ��local��Ϊ��
		vector<int> local;	// ��Ű�����voxel�������(��locales�е��±��)����ʱ����
	public:
		voxel()
		{
			Pointindex.clear();
			Z.clear();
			local.clear();
			tag = -1;
			bon = 0; // ��ʼֵ
			time = 0;
		}

	}; // ���ƻ�������

	// �ӿ���
	class local
	{
	public:
		set<int> content; // ����ڲ���voxel���
		set<int> bondary; // ��ű߽��voxel���
		long pointnumber; // ��Ÿ�local�ڵ������
	public:
		local()
		{
			pointnumber = 0;
			content.clear();
			bondary.clear();
		}
		void Clear()
		{
			content.clear();
			pointnumber = 0;
			bondary.clear();
		}
	}; // �ӿ���
} // namespace REC3D

bool FeatureExtraction(cv::Mat &img, cv::Mat &keys, cv::Mat &descs);
int FeatureMatching(cv::Mat &keys1, cv::Mat &descriptors1, cv::Mat &keys2, cv::Mat &descriptors2, cv::Mat &inlierMatchesData, 
				cv::Matx33f &K1, cv::Matx33f &K2, cv::Matx33d &M);
bool FeatureExtractionGPU(cv::Mat &img, cv::Mat &keys, cv::Mat &descs);
int FeatureMatchingGPU(cv::Mat &keys1, cv::Mat &descriptors1, cv::Mat &keys2, cv::Mat &descriptors2, cv::Mat &inlierMatchesData, 
				cv::Matx33f &K1, cv::Matx33f &K2, cv::Matx33d &M);
bool SiftGPUExtractionInilization();
bool SiftGPUMatchingInilization();

void SelectNeighborViews(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud);
void EstimateDepthMap(REC3D::ImageArr &images);
void FilterDepthMap(REC3D::ImageArr &images);
void FuseDepthMap(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud, double sampleProb);
void EstimateColor(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud);
void PointsMeshing(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud, string meshName, float remove_percent, float remove_scale, float dist_insert, float decimate);
void MeshTexturing(REC3D::ImageArr &images, string meshName, string modelPrefix, bool keep_unseen_faces = 1);

bool LoadFromBundler(string bundlerFileName, string cameraFileName, REC3D::ImageArr &images, REC3D::PointCloud &pointCloud);
bool LoadFromColmap(string camerasFileName, string imagesFileName, string pointsFileName, vector<string>& imageNames, REC3D::ImageArr &images, REC3D::PointCloud &pointCloud);
void IterateImageFiles(string pathName, vector<string> &imageNames);
void IterateImageDirs(string pathName, vector<string> &imageDirs);
bool SfM(REC3D::ImageInfos &imageInfos, REC3D::ImagePairMatches &imagePairMatches, REC3D::SparsePoints &sparsePoints, string SfMDir);
void BoundPoints(REC3D::PointCloud &pointCloud, REC3D::PointCloud &pointCloudBound, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax);
void SfMFormatConversion(REC3D::ImageInfos &imageInfos, REC3D::SparsePoints &sparsePoints, REC3D::ImageArr &images, REC3D::PointCloud &pointCloud);
bool SimilarityTransform(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud);
bool SimilarityTransformRansac(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud);

bool CameraBinSave(const string fileName, REC3D::ImageArr &images);
bool CameraBinLoad(const string fileName, REC3D::ImageArr &images);
bool PointBinSave(const string fileName, REC3D::PointCloud &pointCloud);
bool PointBinLoad(const string fileName, REC3D::PointCloud &pointCloud);
bool PLYPointSave(const string fileName, REC3D::PointCloud &pointCloud, int type = 3);
bool PLYPointLoad(const string fileName, REC3D::PointCloud &pointCloud);
bool PLYMeshSave(string fileName, REC3D::Mesh& mesh, int type = 3);
bool PLYMeshLoad(string fileName, REC3D::Mesh& mesh);
bool LidarBinLoad(const string fileName, REC3D::Image &image);

void LoadVobtree(string visual_index_name, bool prepare = 0);
void SaveVobtree(string visual_index_name);
void BuildDistanceMatrix(int imageNum, Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> &locationMatrix);
void IndexImage(unsigned ID, cv::Mat &keys, cv::Mat &descs);
void QueryImage(unsigned ID, vector<int> &query_IDs, cv::Mat &keys, cv::Mat &descs, vector<unsigned> &IDs, 
			map<unsigned, unsigned> &indexIDs, Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> &locationMatrix);
void GetExifInfo(string fileName, double &focalLength, double &longitude, double &latitude, double &altitude);
void LBH2XYZ(double l, double b, double h, double &x, double &y, double &z);
void XYZ2LBH(double &l, double &b, double &h, double x, double y, double z);
void DepthMap2Points(REC3D::Image &image, REC3D::PointCloud &pointCloud, int flag = 0);

void PointsDividing(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud, string dividFolder, int minnumber, int maxnumber, double gridsize);
void LocalMeshing(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud, string meshName, float remove_percent, float remove_scale, float dist_insert);
void Mendholes(REC3D::Mesh &meshpc, REC3D::Mesh &meshs, std::unordered_map<string, REC3D::Mesh::VIndex> &mapVerticess, int col, int row,
			   int minx, int maxx, int miny, int maxy, double voxelsize, vector<REC3D::local> &locales, REC3D::Mesh &mesh, std::unordered_map<string, REC3D::Mesh::VIndex> &mapVerticesg);
void MeshesMerging(int gridsize, float remove_percent, float remove_scale, float decimate, int smooth, string pathName, string meshName);
void PointVoxel(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud, string dividFolder, double minx, double maxx, double miny, double maxy, int minnumber,
				int maxnumber, double gridsize);
bool LocalReconstruct(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud, REC3D::Mesh &meshl, float distInsert = 2.5, bool global = 0,
					  bool bUseFreeSpaceSupport = false, unsigned nItersFixNonManifold = 4, float kSigma = 1.f, float kQual = 1.f, float kb = 4.f,
					  float kf = 3.f, float kRel = 0.1f /*max 0.3*/, float kAbs = 1000.f /*min 500*/, float kOutl = 400.f /*max 700.f*/, float kInf = (float)(INT_MAX / 8));


void SelectNeighborViews2(REC3D::ImageArr &images, REC3D::PointCloud &pointCloud);
void InitDepthMap(REC3D::Image& image);
bool GapInterpolation(REC3D::Image& image);
int EstimateDepthMap2(REC3D::ImageArr& images);