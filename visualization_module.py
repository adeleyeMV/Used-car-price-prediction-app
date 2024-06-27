import pandas as pd
import numpy as np
import streamlit as st
import joblib
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import xgboost
import os 

#import data
# Get the absolute path to the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, 'dataset.csv')

# Load the CSV file using the absolute path
data = pd.read_csv(file_path)

car_makes = [ 'Honda', 'Acura', 'Peugeot', 'Nissan', 'Kia', 'Hyundai', 'Toyota',
        'Ford', 'Lexus', 'Mazda', 'Volkswagen', 'Mercedes-Benz',
        'Land Rover', 'Dodge', 'Subaru', 'Mitsubishi', 'Infiniti', 
        'Volvo', 'Audi', 'Porsche', 'Pontiac', 'BMW', 'Jeep',
        'Chevrolet', 'Suzuki', 'Hummer', 'Iveko']
data = data[data['Make'].isin(car_makes)]  #selct just the common makes in Nigeria


# Home Page
st.set_option('deprecation.showPyplotGlobalUse', False)

def home_page():

    # Welcome message with Markdown styling
    st.markdown(
        """
        <div style='color: #FF4B4B;'>
        <p style='font-size: 23px;  font-weight: bold;'>Welcome to AutoWise!</p>
        </div>
        Are you contemplating selling your car but unsure about the right price?
        Our app provides you with a quick estimate of your car's value.
        Before diving into the predictions, let's explore key factors influencing used car prices in Nigeria.
                
        """,
        unsafe_allow_html=True
    )

 
    st.markdown(
        """
        <div>
        <p></p>
        <p>In Nigeria, the automotive market boasts a diverse range of car makes,
        each contributing to the rich tapestry of choices for car buyers. This visualization explores the distribution
        of car brands in percentage, shedding light on the popularity of various brands in the car market.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    #with st.container(height=400):
    #Distribution of car makes
    car_makes = data['Make'].value_counts().reset_index(name='Count')
    total_records = len(data)
    # Calculate the percentage
    car_makes['Percentage'] = (car_makes['Count'] / total_records) * 100
    car_makes.rename(columns={'index': 'Make'}, inplace=True)
    #car_makes['Brand'] = car_makes['index']
    car_makes = car_makes.sort_values(by='Percentage', ascending=True)

    # Create a bar chart
    #fig_car_makes_distribution = px.bar(car_makes, x='Brand', y='Percentage',
    #                                   color_continuous_scale='green', title='Distribution Of Car Brands (Percentage)')
    
    fig_car_makes_distribution = px.bar(
    car_makes, 
    y='Make',  # Set 'Make' as the y-axis for a horizontal bar plot
    x='Percentage',
    color_continuous_scale='green', 
    orientation='h',  # Set orientation to 'h' for horizontal bars
    title='Distribution Of Car Brands (Percentage)',
    height=550,
    width=600
    )
    # Get the figure's layout and update the title font color
    fig_car_makes_distribution.update_layout( title=dict(text='Distribution Of Car Brands (Percentage)',
                                                    font=dict( size=18)))  # Change 'blue' to the desired color

    # Update y-axis label
    fig_car_makes_distribution.update_yaxes(title_text='Car Brand', title=dict(text='Car Brand', font=dict( size=18)))
    fig_car_makes_distribution.update_xaxes(title_text='Percentage', title=dict(text='Percentage', font=dict( size=18)))
    # Display the plot
    st.plotly_chart(fig_car_makes_distribution)

    #Inference
    st.markdown(
    """
    - **Dominance of Toyota:** Toyota continues to dominate the market,
      representing a significant 44% of the sampled used cars in Nigeria.
      The sustained popularity of Toyota vehicles suggests a strong preference and trust among car buyers.

    - **Diverse Landscape:** The dataset showcases a diverse range of car makes,
      providing a comprehensive snapshot of the Nigerian used car market. 
      From the reliability of Honda to the luxury of Mercedes-Benz, there is a
      wide array of choices available to buyers.

    - **European Influence:** European car brands, including Mercedes-Benz,
      Ford, Volkswagen, and BMW, collectively hold a substantial share. 
      This indicates a notable preference for European engineering and design among Nigerian used car buyers.

    - **Emerging Players:** The presence of emerging brands such as Kia,
      Hyundai, and Mazda suggests a shift in consumer preferences towards
      newer entrants in the automotive market.

    - **Luxury Appeal:** Luxury brands like Range Rover, Mercedes-Benz, Porshe and BMW,
      though not the majority, maintain a strong presence, reflecting a demand for premium
      and high-performance vehicles among used car buyers.

    - **American Icons:** American car brands like Ford, Chevrolet, and Jeep contribute to
      the diverse mix, highlighting the enduring popularity of American vehicles in Nigeria.

    These insights provide valuable perspectives for understanding and navigating
    the dynamic landscape of the Nigerian used car market. Whether you prioritize 
    reliability, luxury, or emerging trends, the market offers a diverse array of choices for every car enthusiast.
    """
    )


    #Average price by brand
    st.markdown("")
    st.markdown("")
    st.markdown(
    """
    **Explore The Diverse World of Car Pricing!**
    The upcoming plot unveils how various car brands influence prices in the automotive market. 
    Get ready to dive into the average prices of cars, brand by brand, and discover the unique stories each one tells 
    in the fascinating realm of automotive costs.
    """
    )

    #with st.container(height=400):
    # Brand-wise Price Comparison Bar Chart
    avg_price_by_make = data.groupby('Make')['Price'].mean().reset_index()
    fig_brand_price = px.bar(avg_price_by_make, x='Make', y='Price', title='Average Price by Brand')
    

    fig_brand_price.update_layout(title=dict(text='Average Price of a Used Car by its Brand',
                                                    font=dict( size=18)))  

    # Update y-axis label
    fig_brand_price.update_yaxes(title_text='Average Price', title=dict(text='Average Price', font=dict( size=18)))
    fig_brand_price.update_xaxes(title_text='Car Brand', title=dict(text='Car Brand', font=dict( size=18)))
    st.plotly_chart(fig_brand_price)

    #inference
    st.markdown(
    """
    - **Luxury Brands Command Premium Prices:**
        - Porsche, Land Rover, and Mercedes-Benz stand out with significantly higher average prices.
        - Suggests a strong link between brand prestige, exclusive features, and resale value.

    - **Mainstream Brands Show Varied Pricing:**
        - Nissan, Peugeot, and Subaru exhibit relatively lower average prices.
        - Factors may include higher production volumes, perceived depreciation rates, and a larger pool of available used cars.

    - **Mid-Range Brands Balance Affordability and Value:**
        - Honda and Toyota fall below the average market price.
        - Reflects a favorable balance between affordability, reliability, and lower maintenance costs.

    - **Diverse Factors Shape Pricing Dynamics:**
        - Brand perception, features, production volume, and market demand contribute to the complex pricing landscape.
        - Consumers can make informed decisions based on preferences and budget considerations.

    - **Insights for Consumers:**
        - The distribution provides valuable insights into the diverse pricing dynamics of the automotive market.
        - Allows consumers to navigate choices based on brand preferences, features, and budget constraints.

    """
    )
    st.markdown("")
    st.markdown("")
    st.markdown("")



    #SubHeader
    st.markdown("<h3 style='font-size: 20px; font-weight: bold; text-align: center;'>Features Influencing The Pricing Of Used Cars In Nigeria</h3>", unsafe_allow_html=True)
    st.markdown(
    """
    Beyond car brands, several factors play a crucial role in determining the pricing of used cars in Nigeria.
    These factors include the car's condition (whether it is foreign used or Nigerian used), mileage (the total
    distance the car has traveled) and the year of manufacture. Join us as we delve into 
    how these features impact the pricing of used cars.
    """
    )

    #with st.container(height=400):
    #price vs condition
    # Create a strip plot for 'condition' on the x-axis and 'price' on the y-axis
    fig_condition_price = px.violin(data, x='Condition', y='Price', color='Condition',
                                labels={'Condition': 'Car Condition', 'Price': 'Price'},
                                title='Distribution of Price by Condition')
    fig_condition_price.update_layout(title=dict(text='Distribution of Price by Condition',
                                                    font=dict( size=20)))  
    
    # Update y-axis label
    fig_condition_price.update_yaxes(title_text='Price', title=dict(text='Price', font=dict( size=18)))
    fig_condition_price.update_xaxes(title_text='Car Condition', title=dict(text='Car Condition', font=dict( size=18)))
    st.plotly_chart(fig_condition_price)
    
    #inference
    st.markdown(
    """
    **Understanding Car Prices by Condition**

    The violin plot unveils interesting trends in the pricing of cars based on their
    condition—Nigerian used or foreign used. A key takeaway is the broader price range 
    observed for foreign used cars, suggesting a more diverse and potentially upscale selection.

    *Foreign Used Cars:*
    Foreign used cars showcase a wider spread in prices, reflecting a spectrum of premium 
    features, conditions, and specifications. This variation caters to buyers seeking a premium 
    driving experience with choices ranging from meticulously maintained vehicles to high-end models. 
    The premium experience associated with foreign used cars is often linked to factors such as low mileage, 
    better maintenance, and advanced features.

    *Nigerian Used Cars:*
    In contrast, Nigerian used cars exhibit a more uniform price distribution, indicating a consistent 
    pricing landscape. This may appeal to budget-conscious buyers looking for reliable transportation 
    within a more predictable and potentially affordable range.

    In essence, the violin plot not only highlights the price differences between foreign and Nigerian 
    used cars but also offers a glimpse into the diverse choices available to car buyers in the Nigerian market.

    """
    )


    #with st.container(height=400):
    #scatter plot price vs mileage
    plt.figure(figsize=(10, 5.5))
    fig_price_mileage = px.scatter(data, x='Mileage', y='Price', title='Price vs Mileage',color='Condition', opacity=0.7)
    fig_price_mileage.update_layout(
        xaxis_title='Mileage',
        yaxis_title='Price',)
    
    fig_price_mileage.update_layout(title=dict(text='Price vs Mileage',
                                                    font=dict( size=20)))  

    # Update y-axis label
    fig_price_mileage.update_yaxes(title_text='Price', title=dict(text='Price', font=dict( size=18)))
    fig_price_mileage.update_xaxes(title_text='Mileage', title=dict(text='Mileage', font=dict( size=18)))
    st.plotly_chart(fig_price_mileage)

    #inference
    st.markdown(
    """
    The scatter plot comparing the mileage and price of foreign used and Nigerian
    used cars reveals an interesting trend. It demonstrates that both foreign and
    Nigerian used cars tend to exhibit higher prices when their mileage is relatively
    low and lower prices when the mileage is higher.

    This observation suggests a common market behavior where cars with lower mileage,
    typically representing newer or less driven vehicles, command higher prices due to
    their perceived better condition and longer potential lifespan. Conversely, as mileage
    increases, the price tends to decrease, reflecting the expected wear and tear associated with higher usage.

    It's important for potential buyers and sellers to consider this trend when assessing
    the value of used cars in the market. Lower mileage can indeed be a valuable selling point,
    impacting the pricing dynamics significantly. This insight underscores the significance of 
    mileage as a key factor in determining the price of both foreign and Nigerian used cars.
    """
    )
 


    # Group by 'Year of manufacture' and calculate the average price
    average_prices = data.groupby('Year of manufacture')['Price'].mean().reset_index()

    #with st.container(height=400):
    # Create a line plot
    fig_price_yom = px.line(average_prices, x='Year of manufacture', y='Price', title='Average Price vs Year of manufacture')
    fig_price_yom.update_layout(
        xaxis_title='Year of manufacture',
        yaxis_title='Average Price',
    )
    fig_price_yom.update_layout(title=dict(text='Price vs Year of manufacture',
                                                    font=dict( size=20)))  

    # Update y-axis label
    fig_price_yom.update_yaxes(title_text='Price', title=dict(text='Price', font=dict( size=18)))
    fig_price_yom.update_xaxes(title_text='Year of manufacture', title=dict(text='Year of manufacture', font=dict( size=18)))
    st.plotly_chart(fig_price_yom)

    # Inference
    st.markdown(
        """
        Unveiling the dynamics between 'Year of Manufacture' and 'Price' through the engaging
        scatter plot reveals a compelling narrative. The discernible upward trajectory in car
        prices as the manufacturing year advances speaks to a robust correlation between these
        factors. This trend signifies a premium associated with newer vehicles in the thriving used car market. 

        The rationale behind this phenomenon extends beyond mere chronology. Modern cars, born in
        more recent years, are often perceived as embodying cutting-edge features, advanced technology,
        heightened safety standards, and diminished wear and tear. This collective enhancement contributes
        to a buyer's willingness to invest more, drawn by the allure of acquiring a vehicle that encapsulates
        the latest innovations and a sense of prestige.

        As consumers navigate the used car landscape, this insight becomes pivotal, empowering them to make
        informed decisions based on a nuanced understanding of how the interplay of manufacturing years and
        pricing unfolds.
        """
    )


    # Conclusion
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown(
        """

        Our visualization journey provides a comprehensive look into the factors influencing used car prices in Nigeria. 
        From the distribution of car makes to brand-wise price comparisons, we've unraveled the intricacies of the market. 
        For personalized predictions on your car's value, head to the **Prediction Page**. Input your car's details, and let 
        our model estimate its price based on the patterns we've uncovered.

        *Thank you for exploring with us, and happy car hunting!*
        """
    )



