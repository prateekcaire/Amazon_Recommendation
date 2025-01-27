import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card';
import { Star } from 'lucide-react';

const ProductCard = ({ product }) => {
  const stars = Array(5).fill(0).map((_, index) => (
    <Star
      key={index}
      size={16}
      className={index < Math.floor(product.rating) ? 'text-yellow-400' : 'text-gray-300'}
      fill={index < Math.floor(product.rating) ? 'currentColor' : 'none'}
    />
  ));

  return (
    <Card className="w-48 m-2">
      <CardHeader className="p-2">
        <img
          src={product.image_url || "/api/placeholder/200/200"}
          alt={product.title}
          className="w-full h-32 object-cover rounded"
        />
      </CardHeader>
      <CardContent className="p-2">
        <h3 className="font-medium text-sm truncate">{product.title}</h3>
        <div className="flex items-center mt-1">
          {stars}
          <span className="ml-1 text-sm text-gray-600">({product.rating})</span>
        </div>
      </CardContent>
      <CardFooter className="p-2">
        <span className="text-lg font-bold">${product.price}</span>
      </CardFooter>
    </Card>
  );
};

const CategoryRow = ({ category, products }) => (
  <div className="mb-8">
    <h2 className="text-2xl font-bold mb-4 px-4">{category}</h2>
    <div className="flex overflow-x-auto px-4">
      {products.map((product) => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  </div>
);

const RecommendationPage = ({ userId = 0 }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        const response = await fetch(`/api/recommendations/${userId}`);
        if (!response.ok) throw new Error('Failed to fetch recommendations');
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [userId]);

  if (loading) return <div className="p-8 text-center">Loading recommendations...</div>;
  if (error) return <div className="p-8 text-center text-red-500">Error: {error}</div>;
  if (!data) return null;

  const { recommendations, metadata } = data;
  const getProductData = (productId) => ({
    id: productId,
    ...metadata.products[productId]
  });

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 px-4">Recommended for You</h1>
        {Object.entries(recommendations).map(([categoryId, productIds]) => (
          <CategoryRow
            key={categoryId}
            category={metadata.categories[categoryId]}
            products={productIds.map(getProductData)}
          />
        ))}
      </div>
    </div>
  );
};

export default RecommendationPage;