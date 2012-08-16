#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <queue>

const double PI = 3.14159265358979323846;
const double INF = 1e20;
const double EPS = 1e-6;
const double MaxDepth = 5;

// *** その他の関数 ***
inline double rand01() { return (double)rand()/RAND_MAX; }

// *** データ構造 ***
struct Vec {
	double x, y, z;
	Vec(const double x_ = 0, const double y_ = 0, const double z_ = 0) : x(x_), y(y_), z(z_) {}
	inline Vec operator+(const Vec &b) const {return Vec(x + b.x, y + b.y, z + b.z);}
	inline Vec operator-(const Vec &b) const {return Vec(x - b.x, y - b.y, z - b.z);}
	inline Vec operator*(const double b) const {return Vec(x * b, y * b, z * b);}
	inline Vec operator/(const double b) const {return Vec(x / b, y / b, z / b);}
	inline const double LengthSquared() const { return x*x + y*y + z*z; }
	inline const double Length() const { return sqrt(LengthSquared()); }
};
inline Vec operator*(double f, const Vec &v) { return v * f; }
inline Vec Normalize(const Vec &v) { return v / v.Length(); }
// 要素ごとの積をとる
inline const Vec Multiply(const Vec &v1, const Vec &v2) {
	return Vec(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
inline const double Dot(const Vec &v1, const Vec &v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline const Vec Cross(const Vec &v1, const Vec &v2) {
	return Vec((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x));
}
typedef Vec Color;
const Color BackgroundColor(0.0, 0.0, 0.0);

struct Ray {
	Vec org, dir;
	Ray(const Vec org_, const Vec &dir_) : org(org_), dir(dir_) {}
};

enum ReflectionType {
	DIFFUSE,    // 完全拡散面。いわゆるLambertian面。
	SPECULAR,   // 理想的な鏡面。
	REFRACTION, // 理想的なガラス的物質。
};

struct Sphere {
	double radius;
	Vec position;
	Color emission, color;
	ReflectionType ref_type;

	Sphere(const double radius_, const Vec &position_, const Color &emission_, const Color &color_, const ReflectionType ref_type_) :
	  radius(radius_), position(position_), emission(emission_), color(color_), ref_type(ref_type_) {}
	// 入力のrayに対する交差点までの距離を返す。交差しなかったら0を返す。
	const double intersect(const Ray &ray) {
		Vec o_p = position - ray.org;
		const double b = Dot(o_p, ray.dir), det = b * b - Dot(o_p, o_p) + radius * radius;
		if (det >= 0.0) {
			const double sqrt_det = sqrt(det);
			const double t1 = b - sqrt_det, t2 = b + sqrt_det;
			if (t1 > EPS)		return t1;
			else if(t2 > EPS)	return t2;
		}
		return 0.0;
	}
};

// 以下フォトンマップ用データ構造
struct Photon {
	Vec position;
	Color power;
	Vec incident;

	Photon(const Vec& position_, const Color& power_, const Vec& incident_) :
	position(position_), power(power_), incident(incident_) {}
};

// PhotonQueueに乗せるためのデータ構造。struct Photonとは別に用意する。
struct PhotonForQueue {
	const Photon *photon;
	double distance2;
	PhotonForQueue(const Photon *photon_, const double distance2_) : photon(photon_), distance2(distance2_) {}
	bool operator<(const PhotonForQueue &b) const {
		return distance2 < b.distance2;
	}
};
// k-NN searchで使うキュー
typedef std::priority_queue<PhotonForQueue, std::vector<PhotonForQueue>> PhotonQueue;

// フォトンを格納するためのKD-tree
class PhotonMap {
public:
	// k-NN searchのクエリ
	struct Query {
		double max_distance2; // 探索の最大半径
		size_t max_photon_num; // 最大フォトン数
		Vec search_position; // 探索中心
		Vec normal; // 探索中心における法線
		Query(const Vec &search_position_, const Vec &normal_, const double max_distance2_, const size_t max_photon_num_) :
		max_distance2(max_distance2_), normal(normal_), max_photon_num(max_photon_num_), search_position(search_position_) {}
	};
private:
	std::vector<Photon> photons;
	struct KDTreeNode {
		Photon* photon;
		KDTreeNode* left;
		KDTreeNode* right;
		int axis;
	};
	KDTreeNode* root;
	void delete_kdtree(KDTreeNode* node) {
		if (node == NULL)
			return;
		delete_kdtree(node->left);
		delete_kdtree(node->right);
		delete node;
	}

	// フツーのk-NN search。
	void locate_photons(PhotonQueue* pqueue, KDTreeNode* node, PhotonMap::Query &query) {
		if (node == NULL)
			return;
		const int axis = node->axis;

		double delta;
		switch (axis) {
		case 0: delta = query.search_position.x - node->photon->position.x; break;
		case 1: delta = query.search_position.y - node->photon->position.y; break;
		case 2: delta = query.search_position.z - node->photon->position.z; break;
		}

		// フォトン<->探索中心の距離が設定半径以下　かつ　フォトン<->探索中心の法線方向の距離が一定以下　という条件ならそのフォトンを格納
		const Vec dir = node->photon->position - query.search_position;
		const double distance2 = dir.LengthSquared();
		const double dt = Dot(query.normal, dir / sqrt(distance2));
		if (distance2 < query.max_distance2 && fabs(dt) <= query.max_distance2 * 0.01) {
			pqueue->push(PhotonForQueue(node->photon, distance2));
			if (pqueue->size() > query.max_photon_num) {
				pqueue->pop();
				query.max_distance2 = pqueue->top().distance2;
			}
		}
		if (delta > 0.0) { // みぎ
			locate_photons(pqueue,node->right, query);
			if (delta * delta < query.max_distance2) {
				locate_photons(pqueue, node->left, query);
			}
		} else { // ひだり
			locate_photons(pqueue,node->left, query);
			if (delta * delta < query.max_distance2) {
				locate_photons(pqueue, node->right, query);
			}
		}

	}
	
	static bool kdtree_less_operator_x(const Photon& left, const Photon& right) {
		return left.position.x < right.position.x;
	}
	static bool kdtree_less_operator_y(const Photon& left, const Photon& right) {
		return left.position.y < right.position.y;
	}
	static bool kdtree_less_operator_z(const Photon& left, const Photon& right) {
		return left.position.z < right.position.z;
	}
	
	KDTreeNode* create_kdtree_sub(std::vector<Photon>::iterator begin, std::vector<Photon>::iterator end, int depth) {
		if (end - begin <= 0) {
			return NULL;
		}
		const int axis = depth % 3;
		// 中央値
		switch (axis) {
		case 0: std::sort(begin, end, kdtree_less_operator_x); break;
		case 1: std::sort(begin, end, kdtree_less_operator_y); break;
		case 2: std::sort(begin, end, kdtree_less_operator_z); break;
		}
		const int median = (end - begin) / 2;
		KDTreeNode* node = new KDTreeNode;
		node->axis = axis;
		node->photon = &(*(begin + median));
		// 子供
		node->left  = create_kdtree_sub(begin,              begin + median, depth + 1);
		node->right = create_kdtree_sub(begin + median + 1, end,            depth + 1);
		return node;
	}
public:
	PhotonMap() {
		root = NULL;
	}
	virtual ~PhotonMap() {
		delete_kdtree(root);
	}
	size_t Size() {
		return photons.size();
	}
	void SearchKNN(PhotonQueue* pqueue, PhotonMap::Query &query) {
		locate_photons(pqueue, root, query);
	}
	void AddPhoton(const Photon &photon) {
		photons.push_back(photon);
	}
	void CreateKDtree() {
		root = create_kdtree_sub(photons.begin(), photons.end(), 0);
	}
};

// 以下プログレッシブフォトンマップ用データ構造
struct Hitpoint {
	Vec position;
	Vec normal;
	Vec incident;
	int id;
	int image_index;

	double weight;
	float R;
	int N;
	Vec accumulated_color;
	Vec emission_color;

	Hitpoint(const Vec &position_, const Vec &normal_, const Vec &incident_, const int id_, const int image_index_, const double weight_, const Vec &emission_color_) :
	position(position_), normal(normal_), incident(incident_), id(id_), image_index(image_index_), weight(weight_), emission_color(emission_color_) {
		N = 0;
		R = 5.0;
	}
};

// *** レンダリングするシーンデータ ****
// from smallpt
Sphere spheres[] = {
	Sphere(5.0, Vec(50.0, 75.0, 81.6),Color(12,12,12), Color(), DIFFUSE),//照明
	Sphere(1e5, Vec( 1e5+1,40.8,81.6), Color(), Color(0.75, 0.25, 0.25),DIFFUSE),// 左
	Sphere(1e5, Vec(-1e5+99,40.8,81.6),Color(), Color(0.25, 0.25, 0.75),DIFFUSE),// 右
	Sphere(1e5, Vec(50,40.8, 1e5),     Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 奥
	Sphere(1e5, Vec(50,40.8,-1e5+170), Color(), Color(), DIFFUSE),// 手前
	Sphere(1e5, Vec(50, 1e5, 81.6),    Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 床
	Sphere(1e5, Vec(50,-1e5+81.6,81.6),Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 天井
	Sphere(16.5,Vec(27,16.5,47),       Color(), Color(1,1,1)*.99, SPECULAR),// 鏡
	Sphere(16.5,Vec(73,16.5,78),       Color(), Color(1,1,1)*.99, REFRACTION),//ガラス
};
const int LightID = 0;

// *** レンダリング用関数 ***
// シーンとの交差判定関数
inline bool intersect_scene(const Ray &ray, double *t, int *id) {
	const double n = sizeof(spheres) / sizeof(Sphere);
	*t  = INF;
	*id = -1;
	for (int i = 0; i < int(n); i ++) {
		double d = spheres[i].intersect(ray);
		if (d > 0.0 && d < *t) {
			*t  = d;
			*id = i;
		}
	}
	return *t < INF;
}

// プログレッシブフォトンマップの前処理。
// シーンを普通に（パストレ等のように）追跡する。拡散面にあたったらその位置をHitpointに保存する。
// weightはロシアンルーレットの確率等を保存しておく（あとで使う）。
void trace_scene(const Ray &ray, const int image_index, std::vector<Hitpoint> &hitpoints, const double weight, const int depth) {
	double t; // レイからシーンの交差位置までの距離
	int id;   // 交差したシーン内オブジェクトのID
	if (!intersect_scene(ray, &t, &id))
		return;

	const Sphere &obj = spheres[id];
	const Vec hitpoint = ray.org + t * ray.dir; // 交差位置
	const Vec normal  = Normalize(hitpoint - obj.position); // 交差位置の法線
	const Vec orienting_normal = Dot(normal, ray.dir) < 0.0 ? normal : (-1.0 * normal); // 交差位置の法線（物体からのレイの入出を考慮）

	// 色の反射率最大のものを得る。ロシアンルーレットで使う。
	// ロシアンルーレットの閾値は任意だが色の反射率等を使うとより良い。
	double russian_roulette_probability = std::max(obj.color.x, std::max(obj.color.y, obj.color.z));
	// 一定以上レイを追跡したらロシアンルーレットを実行し追跡を打ち切るかどうかを判断する
	if (depth > MaxDepth) {
		if (rand01() >= russian_roulette_probability)
			return;
	} else
		russian_roulette_probability = 1.0; // ロシアンルーレット実行しなかった

	switch (obj.ref_type) {
	case DIFFUSE: {
		hitpoints.push_back(Hitpoint(hitpoint, normal, ray.dir, id, image_index, weight / russian_roulette_probability, obj.emission));
		return;
	} break;

	case SPECULAR: {
		// 完全鏡面にヒットした場合、反射方向から放射輝度をもらってくる
		trace_scene(Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir)), image_index, hitpoints, weight / russian_roulette_probability, depth+1);
		return;
	} break;
	case REFRACTION: {
		Ray reflection_ray = Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir));
		bool into = Dot(normal, orienting_normal) > 0.0; // レイがオブジェクトから出るのか、入るのか

		// Snellの法則
		const double nc = 1.0; // 真空の屈折率
		const double nt = 1.5; // オブジェクトの屈折率
		const double nnt = into ? nc / nt : nt / nc;
		const double ddn = Dot(ray.dir, orienting_normal);
		const double cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);
		
		if (cos2t < 0.0) { // 全反射した	
			// 反射方向から放射輝度をもらってくる
			trace_scene(reflection_ray, image_index, hitpoints, weight / russian_roulette_probability, depth+1);
			return;
		}
		// 屈折していく方向
		Vec tdir = Normalize(ray.dir * nnt - normal * (into ? 1.0 : -1.0) * (ddn * nnt + sqrt(cos2t)));

		// SchlickによるFresnelの反射係数の近似
		const double a = nt - nc, b = nt + nc;
		const double R0 = (a * a) / (b * b);
		const double c = 1.0 - (into ? -ddn : Dot(tdir, normal));
		const double Re = R0 + (1.0 - R0) * pow(c, 5.0);
		const double Tr = 1.0 - Re; // 屈折光の運ぶ光の量
		const double probability  = 0.25 + 0.5 * Re;
		
		// 一定以上レイを追跡したら屈折と反射のどちらか一方を追跡する。
		// ロシアンルーレットで決定する。
		if (depth > 2) {
			if (rand01() < probability) { // 反射
				trace_scene(reflection_ray, image_index, hitpoints, Re * weight / probability / russian_roulette_probability, depth+1);
				return;
			} else { // 屈折
				trace_scene(Ray(hitpoint, tdir), image_index, hitpoints, Tr * weight / (1.0 - probability) / russian_roulette_probability, depth+1);
				return;
			}
		} else { // 屈折と反射の両方を追跡
			trace_scene(reflection_ray, image_index, hitpoints, Re * weight / russian_roulette_probability, depth+1);
			trace_scene(Ray(hitpoint, tdir), image_index, hitpoints, Tr * weight / russian_roulette_probability, depth+1);
			return;
		}
	} break;
	}

}

// フォトン追跡法によりフォトンマップ構築
void create_photon_map(const int shoot_photon_num, PhotonMap *photon_map) {
	std::cout << "Shooting photons... (" << shoot_photon_num << " photons)" << std::endl;
	for (int i = 0; i < shoot_photon_num; i ++) {
		// 光源からフォトンを発射する
		// 光源上の一点をサンプリングする	
		const double r1 = 2 * PI * rand01();
		const double r2 = 1.0 - 2.0 * rand01() ;
		const Vec light_pos = spheres[LightID].position + ((spheres[LightID].radius + EPS) * Vec(sqrt(1.0 - r2*r2) * cos(r1), sqrt(1.0 - r2*r2) * sin(r1), r2));

		const Vec normal = Normalize(light_pos - spheres[LightID].position);
		// 光源上の点から半球サンプリングする
		Vec w, u, v;
		w = normal;
		if (fabs(w.x) > 0.1)
			u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
		else
			u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
		v = Cross(w, u);
		// コサイン項に比例させる。フォトンが運ぶのが放射輝度ではなく放射束であるため。
		const double u1 = 2 * PI * rand01();
		const double u2 = rand01(), u2s = sqrt(u2);
		Vec light_dir = Normalize((u * cos(u1) * u2s + v * sin(u1) * u2s + w * sqrt(1.0 - u2)));

		Ray now_ray(light_pos, light_dir);
		// emissionの値は放射輝度だが、フォトンが運ぶのは放射束なので変換する必要がある。
		// L（放射輝度）= dΦ/(cosθdωdA)なので、光源の放射束はΦ = ∫∫L・cosθdωdAになる。今回は球光源で完全拡散光源であることから
		// 球上の任意の場所、任意の方向に等しい放射輝度Leを持つ。（これがemissionの値）よって、
		// Φ = Le・∫∫cosθdωdAで、Le・∫dA∫cosθdωとなり、∫dAは球の面積なので4πr^2、∫cosθdωは立体角の積分なのでπとなる。
		// よって、Φ = Le・4πr^2・πとなる。この値を光源から発射するフォトン数で割ってやれば一つのフォトンが運ぶ放射束が求まる。
		Color now_flux = spheres[LightID].emission * 4.0 * PI * pow(spheres[LightID].radius, 2.0) * PI / shoot_photon_num;

		// フォトンがシーンを飛ぶ
		bool trace_end = false;
		for (;!trace_end;) {
			// 放射束が0.0なフォトンを追跡してもしょうがないので打ち切る
			if (std::max(now_flux.x, std::max(now_flux.y, now_flux.z)) <= 0.0)
				break;

			double t; // レイからシーンの交差位置までの距離
			int id;   // 交差したシーン内オブジェクトのID
			if (!intersect_scene(now_ray, &t, &id))
				break;
			const Sphere &obj = spheres[id];
			const Vec hitpoint = now_ray.org + t * now_ray.dir; // 交差位置
			const Vec normal  = Normalize(hitpoint - obj.position); // 交差位置の法線
			const Vec orienting_normal = Dot(normal, now_ray.dir) < 0.0 ? normal : (-1.0 * normal); // 交差位置の法線（物体からのレイの入出を考慮）

			switch (obj.ref_type) {
			case DIFFUSE: {
				// 拡散面なのでフォトンをフォトンマップに格納する
				photon_map->AddPhoton(Photon(hitpoint, now_flux, now_ray.dir));

				// 反射するかどうかをロシアンルーレットで決める
				// 例によって確率は任意。今回はフォトンマップ本に従ってRGBの反射率の平均を使う
				const double probability = (obj.color.x + obj.color.y + obj.color.z) / 3;
				if (probability > rand01()) { // 反射
					// orienting_normalの方向を基準とした正規直交基底(w, u, v)を作る。この基底に対する半球内で次のレイを飛ばす。
					Vec w, u, v;
					w = orienting_normal;
					if (fabs(w.x) > 0.1)
						u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
					else
						u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
					v = Cross(w, u);
					// コサイン項を使った重点的サンプリング
					const double r1 = 2 * PI * rand01();
					const double r2 = rand01(), r2s = sqrt(r2);
					Vec dir = Normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1.0 - r2)));
					
					now_ray = Ray(hitpoint, dir);
					now_flux = Multiply(now_flux, obj.color) / probability;
					continue;
				} else { // 吸収（すなわちここで追跡終了）
					trace_end = true;
					continue;
				}
			} break;
			case SPECULAR: {
				// 完全鏡面なのでフォトン格納しない
				// 完全鏡面なのでレイの反射方向は決定的。
				now_ray = Ray(hitpoint, now_ray.dir - normal * 2.0 * Dot(normal, now_ray.dir));
				now_flux = Multiply(now_flux, obj.color);
				continue;
			} break;
			case REFRACTION: {
				// やはりフォトン格納しない
				Ray reflection_ray = Ray(hitpoint, now_ray.dir - normal * 2.0 * Dot(normal, now_ray.dir));
				bool into = Dot(normal, orienting_normal) > 0.0; // レイがオブジェクトから出るのか、入るのか

				// Snellの法則
				const double nc = 1.0; // 真空の屈折率
				const double nt = 1.5; // オブジェクトの屈折率
				const double nnt = into ? nc / nt : nt / nc;
				const double ddn = Dot(now_ray.dir, orienting_normal);
				const double cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);
		
				if (cos2t < 0.0) { // 全反射した
					now_ray = reflection_ray;
					now_flux = Multiply(now_flux, obj.color);
					continue;
				}
				// 屈折していく方向
				Vec tdir = Normalize(now_ray.dir * nnt - normal * (into ? 1.0 : -1.0) * (ddn * nnt + sqrt(cos2t)));
				const double probability  = 0.5;

				// 屈折と反射のどちらか一方を追跡する。
				// ロシアンルーレットで決定する。
				if (rand01() < probability) { // 反射
					now_ray = Ray(hitpoint, tdir);
					now_flux = Multiply(now_flux, obj.color);
					continue;
				} else { // 屈折
					now_ray = reflection_ray;
					now_flux = Multiply(now_flux, obj.color);
					continue;
				}
			} break;
			}
		}
	}
	std::cout << "Done. (" << photon_map->Size() <<  " photons are stored)" << std::endl;
	std::cout << "Creating KD-tree..." << std::endl;
	photon_map->CreateKDtree();
	std::cout << "Done." << std::endl;
}

// Hitpointの集合からプログレッシブに画像更新
void update_image(const int iteration_num, Color *image, PhotonMap &photon_map, std::vector<Hitpoint> &hitpoints, const double Alpha, const int gahter_max_photon_num) {
	// 各ヒットポイントの更新
	for (int hi = 0; hi < hitpoints.size(); hi ++) {
		PhotonQueue pqueue;
		// k近傍探索。gather_radius半径内のフォトンを最大gather_max_photon_num個集めてくる
		photon_map.SearchKNN(&pqueue, PhotonMap::Query(hitpoints[hi].position, hitpoints[hi].normal, pow(hitpoints[hi].R, 2), gahter_max_photon_num));
			
		// キューからフォトンを取り出しvectorに格納する
		std::vector<const PhotonForQueue> photons;
		photons.reserve(pqueue.size());
		for (;!pqueue.empty();) {
			PhotonForQueue p = pqueue.top(); pqueue.pop();
			photons.push_back(p);
		}

		if (hitpoints[hi].N == 0) { // まだフォトンが一個も集められていなかった場合だけ入る
			if (photons.size() > 0) {
				// 初期値設定
				hitpoints[hi].N = photons.size();
				hitpoints[hi].R = sqrt(photons[0].distance2);
				for (int k = 0; k < photons.size(); k ++) {
					const Color v = Multiply(spheres[hitpoints[hi].id].color, photons[k].photon->power) / PI; // Diffuse面のBRDF = 1.0 / πであったのでこれをかける
					hitpoints[hi].accumulated_color = hitpoints[hi].accumulated_color + v;
				}
			}
		} else { // 二回目以降に入る
			// 以下がプログレッシブフォトンマップのキモ
			// 半径を狭める処理
			const int N = hitpoints[hi].N;
			const int M = photons.size();
			const double newR = hitpoints[hi].R * sqrt((N + Alpha * M) / (N + M));
			
			// 新しく追加されたフォトンの寄与計算
			Color tauM;
			for (int k = 0; k < photons.size(); k ++) {
				const Color v = Multiply(spheres[hitpoints[hi].id].color, photons[k].photon->power) / PI; // Diffuse面のBRDF = 1.0 / πであったのでこれをかける
				tauM = tauM + v;
			}
			const Color newtau = (hitpoints[hi].accumulated_color + tauM) * ((N + Alpha * M) / (N + M));

			// 更新
			hitpoints[hi].N = N + Alpha * M;
			hitpoints[hi].R = newR;
			hitpoints[hi].accumulated_color = newtau;
		}

		// フォトンが一個でも集められていたら放射輝度推定する
		if (hitpoints[hi].N > 0) {
			image[hitpoints[hi].image_index] = image[hitpoints[hi].image_index] +
			hitpoints[hi].emission_color + hitpoints[hi].weight * hitpoints[hi].accumulated_color / (PI * pow(hitpoints[hi].R, 2)) / (iteration_num + 1);
		}
	}
}




// *** .hdrフォーマットで出力するための関数 ***
struct HDRPixel {
	unsigned char r, g, b, e;
	HDRPixel(const unsigned char r_ = 0, const unsigned char g_ = 0, const unsigned char b_ = 0, const unsigned char e_ = 0) :
	r(r_), g(g_), b(b_), e(e_) {};
	unsigned char get(int idx) {
		switch (idx) {
		case 0: return r;
		case 1: return g;
		case 2: return b;
		case 3: return e;
		} return 0;
	}

};

// doubleのRGB要素を.hdrフォーマット用に変換
HDRPixel get_hdr_pixel(const Color &color) {
	double d = std::max(color.x, std::max(color.y, color.z));
	if (d <= 1e-32)
		return HDRPixel();
	int e;
	double m = frexp(d, &e); // d = m * 2^e
	d = m * 256.0 / d;
	return HDRPixel(color.x * d, color.y * d, color.z * d, e + 128);
}

// 書き出し用関数
void save_hdr_file(const std::string &filename, const Color* image, const int width, const int height) {
	FILE *fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		std::cerr << "Error: " << filename << std::endl;
		return;
	}
	// .hdrフォーマットに従ってデータを書きだす
	// ヘッダ
	unsigned char ret = 0x0a;
	fprintf(fp, "#?RADIANCE%c", (unsigned char)ret);
	fprintf(fp, "# Made with 100%% pure HDR Shop%c", ret);
	fprintf(fp, "FORMAT=32-bit_rle_rgbe%c", ret);
	fprintf(fp, "EXPOSURE=1.0000000000000%c%c", ret, ret);

	// 輝度値書き出し
	fprintf(fp, "-Y %d +X %d%c", height, width, ret);
	for (int i = height - 1; i >= 0; i --) {
		std::vector<HDRPixel> line;
		for (int j = 0; j < width; j ++) {
			HDRPixel p = get_hdr_pixel(image[j + i * width]);
			line.push_back(p);
		}
		fprintf(fp, "%c%c", 0x02, 0x02);
		fprintf(fp, "%c%c", (width >> 8) & 0xFF, width & 0xFF);
		for (int i = 0; i < 4; i ++) {
			for (int cursor = 0; cursor < width;) {
				const int cursor_move = std::min(127, width - cursor);
				fprintf(fp, "%c", cursor_move);
				for (int j = cursor;  j < cursor + cursor_move; j ++)
					fprintf(fp, "%c", line[j].get(i));
				cursor += cursor_move;
			}
		}
	}

	fclose(fp);
}

int main(int argc, char **argv) {
	int width = 320;
	int height = 240;
	int photon_num = 10000;
	double gather_photon_radius = 32.0;
	int gahter_max_photon_num = 65536;
	
	// PPMのパラメータ
	const int iteration = 1000;
	const int output_interval = 10;
	const double Alpha = 0.7; // PPMの重要パラメータ。詳しくは論文。

	// カメラ位置
	Ray camera(Vec(50.0, 52.0, 295.6), Normalize(Vec(0.0, -0.042612, -1.0)));
	// シーン内でのスクリーンのx,y方向のベクトル
	Vec cx = Vec(width * 0.5135 / height);
	Vec cy = Normalize(Cross(cx, camera.dir)) * 0.5135;
	Color *image = new Color[width * height];

	// 一段階目のシーントレース処理
	std::vector<Hitpoint> hitpoints;
	for (int y = 0; y < height; y ++) {
		srand(y * y * y);
		for (int x = 0; x < width; x ++) {
			int image_index = y * width + x;	
			image[image_index] = Color();

			// 2x2のサブピクセルサンプリング
			for (int sy = 0; sy < 2; sy ++) {
				for (int sx = 0; sx < 2; sx ++) {
					// テントフィルターによってサンプリング
					// ピクセル範囲で一様にサンプリングするのではなく、ピクセル中央付近にサンプルがたくさん集まるように偏りを生じさせる
					const double r1 = 2.0 * rand01(), dx = r1 < 1.0 ? sqrt(r1) - 1.0 : 1.0 - sqrt(2.0 - r1);
					const double r2 = 2.0 * rand01(), dy = r2 < 1.0 ? sqrt(r2) - 1.0 : 1.0 - sqrt(2.0 - r2);
					Vec dir = cx * (((sx + 0.5 + dx) / 2.0 + x) / width - 0.5) +
								cy * (((sy + 0.5 + dy) / 2.0 + y) / height- 0.5) + camera.dir;

					// プログレッシブフォトンマップの前処理
					trace_scene(Ray(camera.org + dir * 130.0, Normalize(dir)), image_index, hitpoints, 1.0, 0);
				}
			}
		}
	}

	// 以下、二段階目のプログレッシブ処理

	for (int i = 0; i < iteration; i ++) {
		std::cout << "----- Iteration " << (i + 1) << " -----" << std::endl;
		PhotonMap photon_map;
		create_photon_map(photon_num, &photon_map);

		for (int j = 0; j < width * height; j ++)
			image[j] = Color();

		// プログレッシブに画像更新
		update_image(i, image, photon_map, hitpoints, Alpha, gahter_max_photon_num);
	
		// output_intervalごとに.hdrフォーマットで出力
		if ((i + 1) % output_interval == 0) {
			char fname[256];
			sprintf(fname, "image_%04d.hdr", i + 1);
			save_hdr_file(std::string(fname), image, width, height);
		}
	}

}
