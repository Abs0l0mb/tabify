import { Div } from '@src/classes';

export class CategoriesFormCard extends Div {

    private input: HTMLInputElement;
    private currentValue: number;
    private maxValue: number;
    private editable: boolean;
    
    constructor(
        private categoryData: any,
        initialValue: number,
        maxValue: number,
        parent: Div,
        editable: boolean
    ) {
        super('categories-form-card', parent);

        this.currentValue = initialValue;
        this.maxValue = maxValue;
        this.editable = editable;

        this.render();
    }

    /*
    **
    **
    */
    private render() : void {
        
        const label = document.createElement('div');
        label.className = 'card-label';
        label.textContent = this.categoryData.category_title;
        this.element.appendChild(label);
        
        this.input = document.createElement('input');
        this.input.type = 'number';
        this.input.value = this.currentValue.toString();
        this.input.min = '3';
        this.input.max = this.maxValue.toString();
        
        if (this.editable) {

            this.input.addEventListener('change', (e) => {

                let newVal = parseFloat((e.target as HTMLInputElement).value);
                if (newVal < 3) newVal = 3;
                if (newVal > this.maxValue) newVal = this.maxValue;
                this.input.value = newVal.toString();
                this.currentValue = newVal;
                
                this.emit('update', newVal);
            });

        } else
            this.input.disabled = true;

        this.element.appendChild(this.input);
    }
    
    /*
    **
    **
    */
    public updateValue(newVal: number) : void {

        this.currentValue = newVal;
        this.input.value = newVal.toString();
    }
    
    /*
    **
    **
    */
    public updateMax(newMax: number) : void {

        this.maxValue = newMax;
        this.input.max = newMax.toString();
    }
}
