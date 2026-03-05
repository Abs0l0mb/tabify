import {
	Block,
	Div,
	CategoriesForm 
} from '@src/classes';

export interface ChartOptions {
	editable: boolean;
}

export class Chart extends Div {
	
	private canvas: HTMLCanvasElement;
	private ctx: CanvasRenderingContext2D;
	private categoriesForm: CategoriesForm;
	private categoryTimes: Record<number, number> = {};
	
	constructor(
		private data: Record<number, any>,
		private options: ChartOptions,
		parent: Block
	) {
		
		super('chart', parent);
		
		for (const key in data)
			this.categoryTimes[parseInt(key)] = data[key].totalTime;
		
		this.canvas = document.createElement('canvas');
		this.canvas.width = 150;
		this.canvas.height = 150;
		this.element.appendChild(this.canvas);
		
		const ctx = this.canvas.getContext('2d');

		if (!ctx)
			throw new Error("Impossible d’obtenir le contexte du canvas");
		
		this.ctx = ctx;
		
		this.categoriesForm = new CategoriesForm(
			data,
			this,
			this.categoryTimes,
			this.options.editable
		);
		
		this.categoriesForm.on('update', (updatedTimes: Record<number, number>) => {

			this.categoryTimes = updatedTimes;
			this.drawChart();
			
			this.emit('update', this.categoryTimes);
		});
		
		this.drawChart();
	}
	
	/*
	**
	**
	*/
	private drawChart() : void {

		const ctx = this.ctx;
		const width = this.canvas.width;
		const height = this.canvas.height;
		ctx.clearRect(0, 0, width, height);
		
		let allocated = 0;

		for (const key in this.categoryTimes)
			allocated += this.categoryTimes[key];
		const computedTotal = allocated;
		
		const segments: { label: string; value: number }[] = [];

		for (const key in this.categoryTimes) {
			const categoryId = parseInt(key);
			segments.push({
				label: this.data[categoryId].category_title,
				value: this.categoryTimes[categoryId]
			});
		}
		
		let startAngle = 0;
		const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
		
		segments.forEach((segment, index) => {

			const fraction = segment.value / computedTotal;
			const endAngle = startAngle + fraction * 2 * Math.PI;
			ctx.beginPath();
			ctx.moveTo(width / 2, height / 2);
			ctx.arc(width / 2, height / 2, width / 2, startAngle, endAngle);
			ctx.closePath();
			ctx.fillStyle = colors[index % colors.length];
			ctx.fill();
			
			const midAngle = startAngle + (endAngle - startAngle) / 2;
			const textX = width / 2 + (width / 4) * Math.cos(midAngle);
			const textY = height / 2 + (width / 4) * Math.sin(midAngle);
			ctx.fillStyle = '#000';
			ctx.font = '14px Inter';
			const pct = Math.round(fraction * 100);
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			ctx.fillText(`${pct}%`, textX, textY);
			
			startAngle = endAngle;
		});
	}
	
	/*
	**
	**
	*/
	public updateCategoryTime(categoryId: number, newTime: number) : void {

		this.categoryTimes[categoryId] = newTime;
		this.categoriesForm.updateCategoryTime(categoryId, newTime);
		this.drawChart();
	}
	
	/*
	**
	**
	*/
	public getDataURL() : string {

		return this.canvas.toDataURL();
	}
}
