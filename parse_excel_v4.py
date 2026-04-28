"""
解析QS定量数据Excel文件并重新组织成清晰格式
理解:
- 每个QS类别行: 第0列是类别名,第1-9列是菌株名
- 下一行: 第0列为空,第1-9列是对应的QS值
"""

import pandas as pd

def parse_qs_excel(excel_path, output_path):
    """
    解析Excel文件:
    - QS类别行: 第0行(强qs)、第2行(中qs)、第6行(低qs)、第12行(无qs)
    - 第0列为空的第4、8、10行也包含菌株名和QS值
    """
    df = pd.read_excel(excel_path, header=None)

    result_data = []

    print("开始解析Excel数据...")

    # 找到所有需要处理的行对(菌株名行 + QS值行)
    # 菌株名行索引: 0, 2, 4, 6, 8, 10, 12
    # QS值行索引: 1, 3, 5, 7, 9, 11, 13

    strain_rows = [0, 2, 4, 6, 8, 10, 12]
    qs_rows = [1, 3, 5, 7, 9, 11, 13]

    for i, (strain_row, qs_row) in enumerate(zip(strain_rows, qs_rows)):
        # 确定QS类别
        if strain_row == 0:
            qs_category = '强qs'
        elif strain_row == 2:
            qs_category = '中qs'
        elif strain_row == 6:
            qs_category = '低qs'
        elif strain_row == 12:
            qs_category = '无qs'
        else:
            # 第4、8、10行属于上面的类别
            if strain_row < 6:
                qs_category = '中qs'
            elif strain_row < 12:
                qs_category = '低qs'
            else:
                qs_category = '无qs'

        print(f"\n处理第{strain_row}行(菌株名)和第{qs_row}行(QS值), QS类别: {qs_category}")

        # 遍历每一列(从第1列开始)
        for col in df.columns:
            if col == 0:  # 跳过第0列
                continue

            strain_name = df.iloc[strain_row, col]
            qs_value = df.iloc[qs_row, col]

            # 检查是否有效
            if pd.notna(strain_name) and pd.notna(qs_value):
                strain_name_clean = str(strain_name).strip().lower()
                try:
                    qs_value_float = float(qs_value)
                    result_data.append({
                        'QS类别': qs_category,
                        '菌株名': strain_name_clean,
                        'QS值': qs_value_float,
                        'Excel行号': strain_row
                    })
                    print(f"  {strain_name_clean}: {qs_value_float}")
                except (ValueError, TypeError):
                    print(f"  警告: 无法转换QS值 '{qs_value}' 为浮点数")

    # 创建DataFrame并保存
    if result_data:
        result_df = pd.DataFrame(result_data)
        # 按QS类别和QS值排序
        result_df = result_df.sort_values(['QS类别', 'QS值'], ascending=[True, False])
        result_df = result_df.reset_index(drop=True)
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # 打印统计信息
        print("\n" + "="*60)
        print("Excel解析结果统计")
        print("="*60)
        print(f"总菌株数: {len(result_df)}")

        for qs_category in result_df['QS类别'].unique():
            category_data = result_df[result_df['QS类别'] == qs_category]
            print(f"\n{qs_category}:")
            print(f"  菌株数: {len(category_data)}")
            print(f"  QS值范围: {category_data['QS值'].min():.6f} - {category_data['QS值'].max():.6f}")
            print(f"  QS值均值: {category_data['QS值'].mean():.6f}")

        print(f"\n数据已保存到: {output_path}")
        print("\n所有数据:")
        pd.set_option('display.max_rows', None)
        print(result_df.to_string(index=False))
        pd.reset_option('display.max_rows')

        return result_df
    else:
        print("警告: 未解析到任何数据!")
        return None

if __name__ == '__main__':
    excel_path = r'd:\ZhangHao\给云龙师弟qs定量数据.xlsx'
    output_path = r'd:\ZhangHao\qs_strain_data.csv'

    result_df = parse_qs_excel(excel_path, output_path)
