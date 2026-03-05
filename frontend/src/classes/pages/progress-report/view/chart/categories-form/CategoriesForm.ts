import { 
    Div,
    CategoriesFormCard 
} from '@src/classes';

export class CategoriesForm extends Div {

    private categoryTimes: Record<number, number> = {};
    private cards: Record<number, CategoriesFormCard> = {};
    private editable: boolean;
    
    constructor(
        private data: Record<number, any>,
        parent: Div,
        initialTimes: Record<number, number>,
        editable: boolean
    ) {
        super('categories-form', parent);
        this.editable = editable;
        this.categoryTimes = { ...initialTimes };
        this.renderCards();
    }
    
    /*
    **
    **
    */
    private renderCards() : void {
        
        this.element.innerHTML = '';
        
        for (const key in this.data) {
            const categoryId = parseInt(key, 10);
            
            const card = new CategoriesFormCard(
                this.data[categoryId],
                this.categoryTimes[categoryId],
                this.calculateMaxAvailable(categoryId),
                this,
                this.editable,
            );
            
            card.on('update', (value) => {
                this.handleCardUpdate(categoryId, value)
            });
            
            this.cards[categoryId] = card;
            this.element.appendChild(card.element);
        }
    }
    
    /*
    **
    **
    */
    private calculateMaxAvailable(currentCategory: number): number {

        const maxWorkTime = 160;
        let totalOther = 0;

        for (const key in this.categoryTimes) {
            
            if (parseInt(key, 10) !== currentCategory)
                totalOther += this.categoryTimes[parseInt(key, 10)];
        }
        return maxWorkTime - totalOther;
    }
    
    /*
    **
    **
    */
    private handleCardUpdate(categoryId: number, newTime: number): void {

        this.categoryTimes[categoryId] = newTime;
        
        for (const key in this.cards) {
            const id = parseInt(key, 10);
            this.cards[id].updateMax(this.calculateMaxAvailable(id));
        }
        
        this.emit('update', this.categoryTimes);
    }
    
    /*
    **
    **
    */
    public updateCategoryTime(categoryId: number, newTime: number): void {

        this.categoryTimes[categoryId] = newTime;

        if (this.cards[categoryId])
            this.cards[categoryId].updateValue(newTime);
        
        for (const key in this.cards) {
            const id = parseInt(key, 10);
            this.cards[id].updateMax(this.calculateMaxAvailable(id));
        }
    }
}
