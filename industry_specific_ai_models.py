#!/usr/bin/env python3
"""
Industry-Specific AI Models - Task 23
Specialized AI models for roofing and construction industry
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD")
}

class RoofingMaterialType(Enum):
    """Roofing material types"""
    ASPHALT_SHINGLE = "asphalt_shingle"
    METAL = "metal"
    TILE = "tile"
    SLATE = "slate"
    WOOD_SHAKE = "wood_shake"
    TPO = "tpo"
    EPDM = "epdm"
    MODIFIED_BITUMEN = "modified_bitumen"
    BUILT_UP = "built_up"

class RoofingProjectType(Enum):
    """Types of roofing projects"""
    FULL_REPLACEMENT = "full_replacement"
    REPAIR = "repair"
    INSPECTION = "inspection"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    NEW_CONSTRUCTION = "new_construction"
    RE_ROOF = "re_roof"
    COATING = "coating"

class WeatherCondition(Enum):
    """Weather conditions affecting roofing"""
    HAIL_DAMAGE = "hail_damage"
    WIND_DAMAGE = "wind_damage"
    STORM_DAMAGE = "storm_damage"
    SNOW_LOAD = "snow_load"
    HEAT_DAMAGE = "heat_damage"
    WATER_DAMAGE = "water_damage"

class RoofingEstimator:
    """AI-powered roofing estimation model"""

    def __init__(self):
        # Base costs per square foot (rough estimates)
        self.material_costs = {
            RoofingMaterialType.ASPHALT_SHINGLE: 3.50,
            RoofingMaterialType.METAL: 7.50,
            RoofingMaterialType.TILE: 10.00,
            RoofingMaterialType.SLATE: 15.00,
            RoofingMaterialType.WOOD_SHAKE: 6.00,
            RoofingMaterialType.TPO: 5.50,
            RoofingMaterialType.EPDM: 4.50,
            RoofingMaterialType.MODIFIED_BITUMEN: 4.00,
            RoofingMaterialType.BUILT_UP: 5.00
        }

        # Labor multipliers
        self.complexity_multipliers = {
            "simple": 1.0,    # Single story, low pitch
            "moderate": 1.3,  # Two story, moderate pitch
            "complex": 1.6,   # Multi-story, steep pitch, dormers
            "extreme": 2.0    # Very steep, multiple levels, difficult access
        }

    async def estimate_project(
        self,
        square_feet: float,
        material: RoofingMaterialType,
        project_type: RoofingProjectType,
        complexity: str = "moderate",
        location: Optional[str] = None
    ) -> Dict:
        """Estimate roofing project cost"""
        try:
            # Calculate squares (100 sq ft units)
            squares = square_feet / 100

            # Base material cost
            material_cost_per_sq = self.material_costs.get(material, 5.0)
            base_material_cost = material_cost_per_sq * square_feet

            # Labor cost (typically 50-60% of material cost)
            labor_multiplier = self.complexity_multipliers.get(complexity, 1.3)
            labor_cost = base_material_cost * 0.55 * labor_multiplier

            # Additional costs based on project type
            additional_costs = self._calculate_additional_costs(
                project_type,
                square_feet,
                material
            )

            # Regional adjustment (if location provided)
            regional_multiplier = self._get_regional_multiplier(location)

            # Calculate totals
            subtotal = base_material_cost + labor_cost + additional_costs
            total_cost = subtotal * regional_multiplier

            # Add markup (typically 20-30%)
            markup = total_cost * 0.25
            final_estimate = total_cost + markup

            # Generate detailed breakdown
            breakdown = {
                "square_feet": square_feet,
                "squares": squares,
                "material": material.value,
                "material_cost": round(base_material_cost, 2),
                "labor_cost": round(labor_cost, 2),
                "additional_costs": round(additional_costs, 2),
                "subtotal": round(subtotal, 2),
                "regional_adjustment": round((regional_multiplier - 1) * 100, 1),
                "markup": round(markup, 2),
                "total_estimate": round(final_estimate, 2),
                "cost_per_square_foot": round(final_estimate / square_feet, 2),
                "confidence": self._calculate_confidence(complexity, project_type),
                "estimate_range": {
                    "low": round(final_estimate * 0.85, 2),
                    "high": round(final_estimate * 1.15, 2)
                }
            }

            # Store estimate in database
            await self._store_estimate(breakdown)

            return breakdown

        except Exception as e:
            logger.error(f"Error estimating project: {e}")
            raise

    def _calculate_additional_costs(
        self,
        project_type: RoofingProjectType,
        square_feet: float,
        material: RoofingMaterialType
    ) -> float:
        """Calculate additional costs based on project type"""
        additional = 0

        if project_type == RoofingProjectType.FULL_REPLACEMENT:
            # Tear-off and disposal
            additional += square_feet * 1.00
            # Underlayment and accessories
            additional += square_feet * 0.50

        elif project_type == RoofingProjectType.RE_ROOF:
            # Additional layer complexity
            additional += square_feet * 0.75

        elif project_type == RoofingProjectType.REPAIR:
            # Minimum service charge
            additional = max(500, square_feet * 2.0)

        elif project_type == RoofingProjectType.EMERGENCY:
            # Emergency premium
            additional = square_feet * 3.0

        return additional

    def _get_regional_multiplier(self, location: Optional[str]) -> float:
        """Get cost multiplier based on location"""
        if not location:
            return 1.0

        # Simple regional adjustments (would use real data in production)
        regional_costs = {
            "northeast": 1.15,
            "west": 1.20,
            "midwest": 0.95,
            "south": 0.90,
            "northwest": 1.10,
            "southwest": 1.05
        }

        location_lower = location.lower()
        for region, multiplier in regional_costs.items():
            if region in location_lower:
                return multiplier

        return 1.0

    def _calculate_confidence(
        self,
        complexity: str,
        project_type: RoofingProjectType
    ) -> float:
        """Calculate estimate confidence score"""
        confidence = 0.85  # Base confidence

        # Adjust based on complexity
        if complexity == "simple":
            confidence += 0.10
        elif complexity == "extreme":
            confidence -= 0.15

        # Adjust based on project type
        if project_type in [RoofingProjectType.FULL_REPLACEMENT, RoofingProjectType.RE_ROOF]:
            confidence += 0.05
        elif project_type == RoofingProjectType.EMERGENCY:
            confidence -= 0.10

        return min(max(confidence, 0.5), 0.95)

    async def _store_estimate(self, estimate: Dict):
        """Store estimate in database"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_roofing_estimates (
                    id, square_feet, material, total_estimate,
                    breakdown, confidence, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (
                str(uuid.uuid4()),
                estimate['square_feet'],
                estimate['material'],
                estimate['total_estimate'],
                json.dumps(estimate),
                estimate['confidence']
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing estimate: {e}")

class DamageAssessmentModel:
    """AI model for assessing roof damage"""

    def __init__(self):
        self.damage_patterns = {
            "hail": {
                "indicators": ["dents", "bruising", "exposed mat", "granule loss"],
                "severity_thresholds": {
                    "minor": 10,
                    "moderate": 30,
                    "severe": 60
                }
            },
            "wind": {
                "indicators": ["missing shingles", "lifted edges", "exposed deck", "creased shingles"],
                "severity_thresholds": {
                    "minor": 5,
                    "moderate": 20,
                    "severe": 40
                }
            },
            "water": {
                "indicators": ["stains", "mold", "rot", "sagging"],
                "severity_thresholds": {
                    "minor": 15,
                    "moderate": 35,
                    "severe": 50
                }
            }
        }

    async def assess_damage(
        self,
        damage_type: str,
        affected_area: float,
        total_area: float,
        observations: List[str]
    ) -> Dict:
        """Assess roof damage and recommend action"""
        try:
            # Calculate damage percentage
            damage_percentage = (affected_area / total_area) * 100

            # Determine severity
            severity = self._determine_severity(damage_type, damage_percentage)

            # Calculate repair vs replacement recommendation
            recommendation = self._generate_recommendation(
                severity,
                damage_percentage,
                damage_type
            )

            # Estimate repair cost
            repair_estimate = await self._estimate_repair_cost(
                affected_area,
                severity,
                damage_type
            )

            # Generate report
            assessment = {
                "damage_type": damage_type,
                "affected_area": affected_area,
                "total_area": total_area,
                "damage_percentage": round(damage_percentage, 2),
                "severity": severity,
                "observations": observations,
                "recommendation": recommendation,
                "estimated_cost": repair_estimate,
                "urgency": self._determine_urgency(severity, damage_type),
                "warranty_impact": self._check_warranty_impact(damage_type, severity),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Store assessment
            await self._store_assessment(assessment)

            return assessment

        except Exception as e:
            logger.error(f"Error assessing damage: {e}")
            raise

    def _determine_severity(
        self,
        damage_type: str,
        damage_percentage: float
    ) -> str:
        """Determine damage severity"""
        thresholds = self.damage_patterns.get(
            damage_type,
            {"severity_thresholds": {"minor": 10, "moderate": 30, "severe": 50}}
        )["severity_thresholds"]

        if damage_percentage >= thresholds["severe"]:
            return "severe"
        elif damage_percentage >= thresholds["moderate"]:
            return "moderate"
        elif damage_percentage >= thresholds["minor"]:
            return "minor"
        else:
            return "cosmetic"

    def _generate_recommendation(
        self,
        severity: str,
        damage_percentage: float,
        damage_type: str
    ) -> Dict:
        """Generate repair/replacement recommendation"""
        if severity == "severe" or damage_percentage > 40:
            return {
                "action": "full_replacement",
                "reasoning": "Extensive damage requires complete roof replacement",
                "timeline": "immediate",
                "can_wait": False
            }
        elif severity == "moderate":
            return {
                "action": "major_repair",
                "reasoning": "Significant repairs needed to prevent further damage",
                "timeline": "within_30_days",
                "can_wait": damage_type != "water"
            }
        elif severity == "minor":
            return {
                "action": "minor_repair",
                "reasoning": "Localized repairs will address the issue",
                "timeline": "within_90_days",
                "can_wait": True
            }
        else:
            return {
                "action": "monitor",
                "reasoning": "Cosmetic damage only, monitor for changes",
                "timeline": "next_inspection",
                "can_wait": True
            }

    async def _estimate_repair_cost(
        self,
        affected_area: float,
        severity: str,
        damage_type: str
    ) -> Dict:
        """Estimate repair cost based on damage"""
        # Base repair costs per square foot
        base_costs = {
            "severe": 8.00,
            "moderate": 5.00,
            "minor": 3.00,
            "cosmetic": 1.50
        }

        base_cost = base_costs.get(severity, 4.00)

        # Adjust for damage type
        if damage_type == "water":
            base_cost *= 1.3  # Water damage is more complex
        elif damage_type == "hail":
            base_cost *= 1.1  # Hail damage requires careful repair

        total_cost = affected_area * base_cost

        # Add minimum service charge for small repairs
        if total_cost < 500 and severity != "severe":
            total_cost = 500

        return {
            "estimated_cost": round(total_cost, 2),
            "cost_range": {
                "low": round(total_cost * 0.8, 2),
                "high": round(total_cost * 1.3, 2)
            },
            "includes": [
                "Materials",
                "Labor",
                "Disposal" if severity in ["severe", "moderate"] else None,
                "Warranty" if severity == "severe" else None
            ]
        }

    def _determine_urgency(self, severity: str, damage_type: str) -> str:
        """Determine repair urgency"""
        if severity == "severe":
            return "emergency"
        elif severity == "moderate" and damage_type == "water":
            return "urgent"
        elif severity == "moderate":
            return "scheduled"
        else:
            return "routine"

    def _check_warranty_impact(
        self,
        damage_type: str,
        severity: str
    ) -> Dict:
        """Check warranty implications"""
        if damage_type in ["hail", "wind"]:
            return {
                "covered": True,
                "claim_recommended": severity in ["severe", "moderate"],
                "deductible_applies": True
            }
        else:
            return {
                "covered": severity == "severe",
                "claim_recommended": False,
                "deductible_applies": severity == "severe"
            }

    async def _store_assessment(self, assessment: Dict):
        """Store damage assessment in database"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_damage_assessments (
                    id, damage_type, severity, affected_area,
                    recommendation, assessment_data, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (
                str(uuid.uuid4()),
                assessment['damage_type'],
                assessment['severity'],
                assessment['affected_area'],
                json.dumps(assessment['recommendation']),
                json.dumps(assessment)
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing assessment: {e}")

class MaterialRecommendationEngine:
    """AI engine for recommending roofing materials"""

    def __init__(self):
        self.climate_materials = {
            "hot_humid": [RoofingMaterialType.METAL, RoofingMaterialType.TILE],
            "cold_snowy": [RoofingMaterialType.METAL, RoofingMaterialType.SLATE],
            "moderate": [RoofingMaterialType.ASPHALT_SHINGLE, RoofingMaterialType.METAL],
            "windy": [RoofingMaterialType.METAL, RoofingMaterialType.SLATE],
            "rainy": [RoofingMaterialType.METAL, RoofingMaterialType.SLATE, RoofingMaterialType.TILE]
        }

    async def recommend_material(
        self,
        climate: str,
        budget_per_sqft: float,
        building_type: str = "residential",
        preferences: Optional[List[str]] = None
    ) -> Dict:
        """Recommend optimal roofing material"""
        recommendations = []

        # Get climate-appropriate materials
        climate_suitable = self.climate_materials.get(climate, [])

        # Filter by budget
        estimator = RoofingEstimator()
        for material in RoofingMaterialType:
            material_cost = estimator.material_costs.get(material, 5.0)

            if material_cost <= budget_per_sqft * 0.7:  # Leave room for labor
                score = self._calculate_material_score(
                    material,
                    climate_suitable,
                    building_type,
                    preferences or []
                )

                recommendations.append({
                    "material": material.value,
                    "cost_per_sqft": material_cost,
                    "suitability_score": score,
                    "pros": self._get_material_pros(material),
                    "cons": self._get_material_cons(material),
                    "lifespan_years": self._get_lifespan(material),
                    "warranty_typical": self._get_warranty(material)
                })

        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)

        return {
            "top_recommendation": recommendations[0] if recommendations else None,
            "alternatives": recommendations[1:4] if len(recommendations) > 1 else [],
            "factors_considered": {
                "climate": climate,
                "budget": budget_per_sqft,
                "building_type": building_type,
                "preferences": preferences or []
            }
        }

    def _calculate_material_score(
        self,
        material: RoofingMaterialType,
        climate_suitable: List,
        building_type: str,
        preferences: List[str]
    ) -> float:
        """Calculate material suitability score"""
        score = 50.0  # Base score

        # Climate suitability
        if material in climate_suitable:
            score += 20

        # Building type compatibility
        if building_type == "residential":
            if material in [RoofingMaterialType.ASPHALT_SHINGLE, RoofingMaterialType.METAL]:
                score += 10
        elif building_type == "commercial":
            if material in [RoofingMaterialType.TPO, RoofingMaterialType.EPDM, RoofingMaterialType.METAL]:
                score += 10

        # Preference matching
        for pref in preferences:
            if pref.lower() in material.value.lower():
                score += 5

        return min(score, 100.0)

    def _get_material_pros(self, material: RoofingMaterialType) -> List[str]:
        """Get material advantages"""
        pros = {
            RoofingMaterialType.ASPHALT_SHINGLE: ["Affordable", "Easy installation", "Wide variety"],
            RoofingMaterialType.METAL: ["Durable", "Energy efficient", "Low maintenance"],
            RoofingMaterialType.TILE: ["Long lasting", "Fire resistant", "Aesthetic appeal"],
            RoofingMaterialType.SLATE: ["Extremely durable", "Natural beauty", "Fire resistant"],
            RoofingMaterialType.TPO: ["Energy efficient", "Chemical resistant", "Strong seams"]
        }
        return pros.get(material, ["Good value"])

    def _get_material_cons(self, material: RoofingMaterialType) -> List[str]:
        """Get material disadvantages"""
        cons = {
            RoofingMaterialType.ASPHALT_SHINGLE: ["Shorter lifespan", "Weather vulnerable"],
            RoofingMaterialType.METAL: ["Higher upfront cost", "Can be noisy"],
            RoofingMaterialType.TILE: ["Heavy", "Expensive", "Fragile"],
            RoofingMaterialType.SLATE: ["Very expensive", "Heavy", "Requires expertise"],
            RoofingMaterialType.TPO: ["Newer technology", "Requires professional installation"]
        }
        return cons.get(material, ["Limited availability"])

    def _get_lifespan(self, material: RoofingMaterialType) -> int:
        """Get typical material lifespan in years"""
        lifespans = {
            RoofingMaterialType.ASPHALT_SHINGLE: 20,
            RoofingMaterialType.METAL: 50,
            RoofingMaterialType.TILE: 50,
            RoofingMaterialType.SLATE: 100,
            RoofingMaterialType.TPO: 30,
            RoofingMaterialType.EPDM: 25
        }
        return lifespans.get(material, 25)

    def _get_warranty(self, material: RoofingMaterialType) -> str:
        """Get typical warranty period"""
        warranties = {
            RoofingMaterialType.ASPHALT_SHINGLE: "20-30 years",
            RoofingMaterialType.METAL: "30-50 years",
            RoofingMaterialType.TILE: "50 years",
            RoofingMaterialType.SLATE: "75-100 years",
            RoofingMaterialType.TPO: "20-30 years"
        }
        return warranties.get(material, "25 years")

class IndustrySpecificAIModels:
    """Main industry-specific AI system"""

    def __init__(self):
        self.estimator = RoofingEstimator()
        self.damage_assessor = DamageAssessmentModel()
        self.material_recommender = MaterialRecommendationEngine()

    async def process_roofing_request(
        self,
        request_type: str,
        parameters: Dict
    ) -> Dict:
        """Process industry-specific request"""
        try:
            if request_type == "estimate":
                return await self.estimator.estimate_project(
                    square_feet=parameters['square_feet'],
                    material=RoofingMaterialType[parameters['material'].upper()],
                    project_type=RoofingProjectType[parameters['project_type'].upper()],
                    complexity=parameters.get('complexity', 'moderate'),
                    location=parameters.get('location')
                )

            elif request_type == "damage_assessment":
                return await self.damage_assessor.assess_damage(
                    damage_type=parameters['damage_type'],
                    affected_area=parameters['affected_area'],
                    total_area=parameters['total_area'],
                    observations=parameters.get('observations', [])
                )

            elif request_type == "material_recommendation":
                return await self.material_recommender.recommend_material(
                    climate=parameters['climate'],
                    budget_per_sqft=parameters['budget'],
                    building_type=parameters.get('building_type', 'residential'),
                    preferences=parameters.get('preferences')
                )

            else:
                raise ValueError(f"Unknown request type: {request_type}")

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise

# Singleton instance
_models_instance = None

def get_industry_models():
    """Get or create industry models instance"""
    global _models_instance
    if _models_instance is None:
        _models_instance = IndustrySpecificAIModels()
    return _models_instance

# Export main components
__all__ = [
    'IndustrySpecificAIModels',
    'get_industry_models',
    'RoofingMaterialType',
    'RoofingProjectType',
    'WeatherCondition'
]